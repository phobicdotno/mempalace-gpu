# Apple M1 Benchmark — MPS vs CPU

**Machine:** MacBook M1 (Apple Silicon)  
**Model:** all-MiniLM-L6-v2  
**Date:** 2026-04-09  
**mempalace-gpu:** v3.2.0  

## Test 1: ~/Documents (500 files, 1,239 drawers)

| Metric | MPS (GPU) | CPU Only |
|---|---|---|
| Wall time | 5:48 (348s) | 6:09 (369s) |
| User CPU time | 16.36s | 196.16s |
| System time | 5.52s | 28.43s |
| CPU utilization | 6% | 60% |
| Drawers created | 1,239 | 1,239 |

**Wall-clock speedup: 1.06x** (MPS barely faster)  
**CPU usage reduction: 12x** (MPS frees the CPU)

## Test 2: ~/phobic (500 files, 8,886 drawers)

| Metric | MPS (GPU) | CPU Only |
|---|---|---|
| Wall time | 17:16 (1036s) | 8:28 (508s) |
| User CPU time | 37.54s | 300.37s |
| System time | 18.96s | 54.45s |
| CPU utilization | 5% | 69% |
| Drawers created | 8,886 | 8,886 |

**Wall-clock: CPU is 2.0x faster than MPS**

## Analysis

MPS loses on wall-clock time for mempalace mining because:

1. **Small batch overhead:** The embedding model processes small text chunks. CPU-to-GPU-to-CPU data transfer per batch exceeds GPU compute savings.
2. **I/O bound:** Mining is bottlenecked by file reading, parsing, and ChromaDB writes — not embedding computation.
3. **MPS dispatch latency:** Apple's Metal Performance Shaders have higher dispatch overhead than NVIDIA CUDA for small workloads.

MPS does use significantly less CPU (5-6% vs 60-69%), leaving the processor free for other work. But the 2x wall-time penalty makes CPU the better default.

## Recommendation

- **Apple Silicon mining:** Use `--device cpu` (now the default in auto-detect)
- **NVIDIA GPU mining:** Use `--device cuda` (auto-detected, expected 3-6x speedup)
- **Apple Silicon search queries:** MPS may still help for large concurrent searches (not yet benchmarked)
