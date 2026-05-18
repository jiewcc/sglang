#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


TOPK = 512
DEFAULT_PROVIDERS = ("torch", "flashinfer", "dsv4", "dsv4_v2")
ALL_PROVIDERS = (
    "torch",
    "flashinfer",
    "flashinfer_prepare",
    "flashinfer_core",
    "dsv4",
    "dsv4_v2",
)
DEFAULT_SEQ_LENS = (16_384, 65_536, 98_304, 120_000, 262_144, 524_288, 1_048_576)
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)


@dataclass(frozen=True)
class Case:
    batch_size: int
    max_seq_len: int
    page_size: int
    seq_pattern: str


@dataclass
class Inputs:
    scores: torch.Tensor
    seq_lens: torch.Tensor
    page_tables: torch.Tensor


@dataclass
class BenchRow:
    provider: str
    batch_size: int
    max_seq_len: int
    min_seq_len: int
    page_size: int
    status: str
    median_ms: float | None = None
    p20_ms: float | None = None
    p80_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    speedup_vs_torch: float | None = None
    speedup_vs_dsv4: float | None = None
    note: str = ""


def parse_ints(values: Iterable[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return parsed


def import_providers() -> dict[str, Callable[..., None]]:
    from sglang.jit_kernel.deepseek_v4 import (
        topk_transform_512,
        topk_transform_512_v2,
    )
    from sglang.srt.layers.attention.dsv4.indexer import (
        topk_transform_512_flashinfer,
        topk_transform_512_pytorch_vectorized,
    )

    def run_torch(
        scores: torch.Tensor,
        seq_lens: torch.Tensor,
        page_tables: torch.Tensor,
        out: torch.Tensor,
        page_size: int,
    ) -> None:
        topk_transform_512_pytorch_vectorized(
            scores, seq_lens, page_tables, out, page_size
        )

    def run_flashinfer(
        scores: torch.Tensor,
        seq_lens: torch.Tensor,
        page_tables: torch.Tensor,
        out: torch.Tensor,
        page_size: int,
    ) -> None:
        topk_transform_512_flashinfer(scores, seq_lens, page_tables, out, page_size)

    def run_flashinfer_prepare(
        scores: torch.Tensor,
        seq_lens: torch.Tensor,
        page_tables: torch.Tensor,
        out: torch.Tensor,
        page_size: int,
    ) -> None:
        _ = scores, seq_lens, out
        make_flashinfer_src_page_table(page_tables, scores.shape[1], page_size)

    def run_flashinfer_core(
        scores: torch.Tensor,
        seq_lens: torch.Tensor,
        page_tables: torch.Tensor,
        out: torch.Tensor,
        page_size: int,
        src_page_table: torch.Tensor,
    ) -> None:
        _ = page_tables, page_size
        import flashinfer

        result = flashinfer.top_k_page_table_transform(
            scores,
            src_page_table,
            seq_lens,
            out.shape[1],
            deterministic=False,
        )
        out.copy_(result)

    def run_dsv4(
        scores: torch.Tensor,
        seq_lens: torch.Tensor,
        page_tables: torch.Tensor,
        out: torch.Tensor,
        page_size: int,
    ) -> None:
        topk_transform_512(scores, seq_lens, page_tables, out, page_size)

    def run_dsv4_v2(
        scores: torch.Tensor,
        seq_lens: torch.Tensor,
        page_tables: torch.Tensor,
        out: torch.Tensor,
        page_size: int,
        metadata: torch.Tensor,
    ) -> None:
        topk_transform_512_v2(scores, seq_lens, page_tables, out, page_size, metadata)

    run_dsv4_v2._needs_topk_v2_metadata = True  # type: ignore[attr-defined]
    run_flashinfer_prepare._skip_check = True  # type: ignore[attr-defined]
    run_flashinfer_prepare._skip_speedup = True  # type: ignore[attr-defined]
    run_flashinfer_core._needs_flashinfer_src_page_table = True  # type: ignore[attr-defined]

    return {
        "torch": run_torch,
        "flashinfer": run_flashinfer,
        "flashinfer_prepare": run_flashinfer_prepare,
        "flashinfer_core": run_flashinfer_core,
        "dsv4": run_dsv4,
        "dsv4_v2": run_dsv4_v2,
    }


def make_flashinfer_src_page_table(
    page_tables: torch.Tensor,
    max_seq_len: int,
    page_size: int,
) -> torch.Tensor:
    batch_size = page_tables.shape[0]
    device = page_tables.device
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    positions = torch.arange(max_seq_len, device=device, dtype=torch.int32)
    page_idx = positions >> page_bits
    offset = positions & page_mask
    page_idx_expanded = page_idx.unsqueeze(0).expand(batch_size, -1)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_expanded.long())
    src_page_table = (physical_pages << page_bits) | offset.unsqueeze(0)
    return src_page_table.to(torch.int32)


def make_seq_lens(
    batch_size: int, max_seq_len: int, pattern: str, device: torch.device
) -> torch.Tensor:
    if pattern == "full":
        values = torch.full((batch_size,), max_seq_len, dtype=torch.int32)
    elif pattern == "descending":
        step = max(max_seq_len // max(batch_size, 1), 1)
        values = torch.tensor(
            [max(max_seq_len - i * step, 1) for i in range(batch_size)],
            dtype=torch.int32,
        )
    elif pattern == "mixed":
        fractions = (1.0, 0.75, 0.5, 0.25)
        values = torch.tensor(
            [
                max(int(max_seq_len * fractions[i % len(fractions)]), 1)
                for i in range(batch_size)
            ],
            dtype=torch.int32,
        )
    else:
        raise ValueError(f"unknown seq pattern: {pattern}")
    return values.to(device=device)


def make_inputs(case: Case, seed: int, device: torch.device) -> Inputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + case.batch_size * 1_000_003 + case.max_seq_len)
    scores = torch.randn(
        (case.batch_size, case.max_seq_len),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    seq_lens = make_seq_lens(
        case.batch_size, case.max_seq_len, case.seq_pattern, device
    )

    max_pages = math.ceil(case.max_seq_len / case.page_size)
    logical_pages = torch.arange(max_pages, device=device, dtype=torch.int32)
    page_tables = logical_pages.unsqueeze(0).repeat(case.batch_size, 1)
    batch_offsets = (
        torch.arange(case.batch_size, device=device, dtype=torch.int32).unsqueeze(1)
        * max_pages
    )
    page_tables = page_tables + batch_offsets
    return Inputs(scores=scores, seq_lens=seq_lens, page_tables=page_tables)


def new_output(inputs: Inputs) -> torch.Tensor:
    return torch.empty(
        (inputs.scores.shape[0], TOPK), device=inputs.scores.device, dtype=torch.int32
    )


def valid_sorted(out: torch.Tensor) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for row in out:
        valid = row[row >= 0]
        result.append(torch.sort(valid).values)
    return result


def assert_same_selection(candidate: torch.Tensor, reference: torch.Tensor) -> None:
    cand_rows = valid_sorted(candidate)
    ref_rows = valid_sorted(reference)
    for row_id, (cand, ref) in enumerate(zip(cand_rows, ref_rows)):
        if cand.numel() != ref.numel() or not torch.equal(cand, ref):
            raise AssertionError(
                f"row {row_id}: selected set mismatch "
                f"(candidate={cand[:16].tolist()}, reference={ref[:16].tolist()})"
            )


def timed_run(
    fn: Callable[..., None],
    inputs: Inputs,
    page_size: int,
    warmup_iters: int,
    iters: int,
) -> tuple[float, float, float, float, float]:
    out = new_output(inputs)
    metadata = make_metadata(fn, inputs, page_size)
    for _ in range(warmup_iters):
        call_provider(fn, inputs, out, page_size, metadata)
    torch.cuda.synchronize()

    timings: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        call_provider(fn, inputs, out, page_size, metadata)
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))

    timings.sort()
    return (
        statistics.median(timings),
        percentile(timings, 0.20),
        percentile(timings, 0.80),
        timings[0],
        timings[-1],
    )


def percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def run_provider_once(
    fn: Callable[..., None], inputs: Inputs, page_size: int
) -> torch.Tensor:
    out = new_output(inputs)
    metadata = make_metadata(fn, inputs, page_size)
    call_provider(fn, inputs, out, page_size, metadata)
    torch.cuda.synchronize()
    return out


def make_metadata(
    fn: Callable[..., None], inputs: Inputs, page_size: int
) -> torch.Tensor | None:
    if getattr(fn, "_needs_topk_v2_metadata", False):
        from sglang.jit_kernel.deepseek_v4 import plan_topk_v2

        return plan_topk_v2(inputs.seq_lens)
    if getattr(fn, "_needs_flashinfer_src_page_table", False):
        return make_flashinfer_src_page_table(
            inputs.page_tables, inputs.scores.shape[1], page_size=page_size
        )
    return None


def call_provider(
    fn: Callable[..., None],
    inputs: Inputs,
    out: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor | None,
) -> None:
    if metadata is None:
        fn(inputs.scores, inputs.seq_lens, inputs.page_tables, out, page_size)
    else:
        fn(inputs.scores, inputs.seq_lens, inputs.page_tables, out, page_size, metadata)


def run_case(
    case: Case,
    provider_names: list[str],
    providers: dict[str, Callable[..., None]],
    seed: int,
    warmup_iters: int,
    iters: int,
    check: bool,
) -> list[BenchRow]:
    device = torch.device("cuda")
    inputs = make_inputs(case, seed=seed, device=device)
    rows: list[BenchRow] = []
    reference: torch.Tensor | None = None

    for provider in provider_names:
        row = BenchRow(
            provider=provider,
            batch_size=case.batch_size,
            max_seq_len=case.max_seq_len,
            min_seq_len=int(inputs.seq_lens.min().item()),
            page_size=case.page_size,
            status="ok",
        )
        fn = providers[provider]
        try:
            if check and not getattr(fn, "_skip_check", False):
                out = run_provider_once(fn, inputs, case.page_size)
                if reference is None:
                    reference = out
                else:
                    assert_same_selection(out, reference)

            row.median_ms, row.p20_ms, row.p80_ms, row.min_ms, row.max_ms = timed_run(
                fn, inputs, case.page_size, warmup_iters, iters
            )
        except Exception as exc:  # noqa: BLE001
            row.status = "skipped"
            row.note = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    by_provider = {row.provider: row for row in rows if row.status == "ok"}
    torch_ms = by_provider.get("torch").median_ms if "torch" in by_provider else None
    dsv4_ms = by_provider.get("dsv4").median_ms if "dsv4" in by_provider else None
    for row in rows:
        if (
            row.status != "ok"
            or row.median_ms is None
            or getattr(providers[row.provider], "_skip_speedup", False)
        ):
            continue
        if torch_ms:
            row.speedup_vs_torch = torch_ms / row.median_ms
        if dsv4_ms:
            row.speedup_vs_dsv4 = dsv4_ms / row.median_ms

    return rows


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_markdown(rows: list[BenchRow]) -> str:
    headers = [
        "provider",
        "B",
        "max_seq_len",
        "min_seq_len",
        "page",
        "status",
        "median_ms",
        "p20_ms",
        "p80_ms",
        "min_ms",
        "max_ms",
        "x_torch",
        "x_dsv4",
        "note",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row.provider,
            str(row.batch_size),
            str(row.max_seq_len),
            str(row.min_seq_len),
            str(row.page_size),
            row.status,
            format_float(row.median_ms),
            format_float(row.p20_ms),
            format_float(row.p80_ms),
            format_float(row.min_ms),
            format_float(row.max_ms),
            format_float(row.speedup_vs_torch, 2),
            format_float(row.speedup_vs_dsv4, 2),
            row.note.replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def truncate_text(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "..."


def render_terminal_table(rows: list[BenchRow], note_width: int = 48) -> str:
    headers = [
        "provider",
        "B",
        "max_len",
        "min_len",
        "page",
        "status",
        "median",
        "p20",
        "p80",
        "x_torch",
        "x_dsv4",
        "note",
    ]
    body = []
    for row in rows:
        body.append(
            [
                row.provider,
                str(row.batch_size),
                str(row.max_seq_len),
                str(row.min_seq_len),
                str(row.page_size),
                row.status,
                format_float(row.median_ms),
                format_float(row.p20_ms),
                format_float(row.p80_ms),
                format_float(row.speedup_vs_torch, 2),
                format_float(row.speedup_vs_dsv4, 2),
                truncate_text(row.note, note_width),
            ]
        )

    widths = [len(header) for header in headers]
    for values in body:
        for i, value in enumerate(values):
            widths[i] = max(widths[i], len(value))

    right_aligned = {
        "B",
        "max_len",
        "min_len",
        "page",
        "median",
        "p20",
        "p80",
        "x_torch",
        "x_dsv4",
    }

    def format_row(values: list[str]) -> str:
        cells = []
        for header, value, width in zip(headers, values, widths):
            if header in right_aligned:
                cells.append(value.rjust(width))
            else:
                cells.append(value.ljust(width))
        return "  ".join(cells)

    lines = [
        format_row(headers),
        format_row(["-" * width for width in widths]),
    ]
    lines.extend(format_row(values) for values in body)
    return "\n".join(lines)


def render_for_terminal(
    rows: list[BenchRow], terminal_format: str, note_width: int
) -> str:
    if terminal_format == "markdown":
        return render_markdown(rows)
    return render_terminal_table(rows, note_width=note_width)


def write_outputs(rows: list[BenchRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dsv4_topk_results.csv"
    md_path = output_dir / "dsv4_topk_results.md"
    fieldnames = list(BenchRow.__dataclass_fields__.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    md_path.write_text(render_markdown(rows) + "\n")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek V4 indexer TopK transform implementations."
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=list(DEFAULT_PROVIDERS),
        choices=list(ALL_PROVIDERS),
        help="Providers to benchmark.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        default=[str(v) for v in DEFAULT_BATCH_SIZES],
        help="Batch sizes, space- or comma-separated.",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        default=[str(v) for v in DEFAULT_SEQ_LENS],
        help="Max sequence lengths, space- or comma-separated.",
    )
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--seq-pattern",
        choices=("full", "descending", "mixed"),
        default="full",
        help="How per-row sequence lengths are generated inside each batch.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip cross-provider selected-set correctness checks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for CSV and Markdown result files.",
    )
    parser.add_argument(
        "--terminal-format",
        choices=("compact", "markdown"),
        default="compact",
        help="Print an aligned terminal table by default, or Markdown for copying.",
    )
    parser.add_argument(
        "--terminal-note-width",
        type=int,
        default=48,
        help="Maximum note column width for compact terminal output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global torch
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    providers = import_providers()
    batch_sizes = parse_ints(args.batch_sizes)
    seq_lens = parse_ints(args.seq_lens)
    cases = [
        Case(
            batch_size=batch_size,
            max_seq_len=seq_len,
            page_size=args.page_size,
            seq_pattern=args.seq_pattern,
        )
        for batch_size in batch_sizes
        for seq_len in seq_lens
    ]

    rows: list[BenchRow] = []
    for case in cases:
        print(
            f"\n== B={case.batch_size}, max_seq_len={case.max_seq_len}, "
            f"page_size={case.page_size}, pattern={case.seq_pattern} =="
        )
        rows.extend(
            run_case(
                case=case,
                provider_names=args.providers,
                providers=providers,
                seed=args.seed,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
                check=not args.no_check,
            )
        )
        print(
            render_for_terminal(
                rows[-len(args.providers) :],
                terminal_format=args.terminal_format,
                note_width=args.terminal_note_width,
            )
        )

    print("\n# DeepSeek V4 TopK Benchmark Results")
    print(
        render_for_terminal(
            rows,
            terminal_format=args.terminal_format,
            note_width=args.terminal_note_width,
        )
    )
    if args.output_dir is not None:
        write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
