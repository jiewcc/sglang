# DeepSeek V4 Indexer TopK Benchmarks

This directory is a small benchmark project for the DeepSeek V4 indexer TopK
transform path in SGLang. It compares the implementations wired in
`python/sglang/srt/layers/attention/dsv4/indexer.py`:

- `torch`: the PyTorch vectorized path using `torch.topk`
- `flashinfer`: end-to-end FlashInfer path, including `src_page_table` creation
- `flashinfer_prepare`: only build the expanded FlashInfer `src_page_table`
- `flashinfer_core`: only call `flashinfer.top_k_page_table_transform` with a
  prebuilt `src_page_table`, then copy the result to `out_page_indices`
- `dsv4`: SGLang's default JIT TopK transform
- `dsv4_v2`: SGLang's planned/workspace JIT TopK transform

The benchmark generates one shared synthetic decode workload per shape:

- `scores`: `[batch_size, max_seq_len]`, `float32`, CUDA
- `seq_lens`: `[batch_size]`, `int32`, CUDA
- `page_tables`: `[batch_size, ceil(max_seq_len / page_size)]`, `int32`, CUDA
- `out_page_indices`: `[batch_size, 512]`, `int32`, CUDA

It runs correctness checks before timing. Because TopK order is not guaranteed
to match across implementations, validation compares the selected index set for
each row after sorting the valid outputs.

## Quick Start

Run from the SGLang repository root:

```bash
python benchmark/kernels/dsv4_topk/bench_dsv4_topk.py
```

Dependencies are expected to come from an SGLang development environment. The
minimal package list is recorded in `requirements.txt`.

Useful long-context run:

```bash
python benchmark/kernels/dsv4_topk/bench_dsv4_topk.py \
  --batch-sizes 1 2 4 \
  --seq-lens 16384 65536 98304 120000 262144 524288 1048576 \
  --providers torch flashinfer dsv4 dsv4_v2 \
  --output-dir benchmark/kernels/dsv4_topk/results
```

To split FlashInfer preparation from the core transform:

```bash
python benchmark/kernels/dsv4_topk/bench_dsv4_topk.py \
  --providers flashinfer flashinfer_prepare flashinfer_core
```

For variable-length batches, make each row shorter than the previous one:

```bash
python benchmark/kernels/dsv4_topk/bench_dsv4_topk.py --seq-pattern descending
```

## Outputs

The script prints a compact aligned table in the terminal and, when
`--output-dir` is set, writes:

- `dsv4_topk_results.csv`
- `dsv4_topk_results.md`

The terminal table reports median, p20, and p80 latency in milliseconds, plus
speedup against `torch` and SGLang default `dsv4` when those providers are part
of the same case. The Markdown/CSV files keep the full min/max latency and full
error notes.

To print Markdown in the terminal instead:

```bash
python benchmark/kernels/dsv4_topk/bench_dsv4_topk.py --terminal-format markdown
```

For compact output, long notes are truncated. Adjust this with:

```bash
python benchmark/kernels/dsv4_topk/bench_dsv4_topk.py --terminal-note-width 96
```

## Notes

- `flashinfer` is optional. If it is unavailable, that provider is reported as
  skipped.
- `dsv4` and `dsv4_v2` require CUDA JIT compilation on the first warmup call.
  Compilation time is excluded from the measured iterations.
- `flashinfer_prepare` is not a TopK provider, so it is timed but skipped during
  correctness checks and speedup comparisons.
- Timings use CUDA events, so they measure GPU work queued by each provider.
  Python CPU launch overhead is not included.
- The PyTorch helper in `indexer.py` is fixed to TopK 512, so this benchmark
  intentionally focuses on `K=512`.
