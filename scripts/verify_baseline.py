#!/usr/bin/env python3
"""
scripts/verify_baseline.py
--------------------------
External Baseline & Numerical Verification for spmv-inspect.
Benchmarks SciPy's optimized C-accelerated CSR SpMV kernel on the same
matrix using identical protocol: 10 warmup + 50 timed iterations (Median ± StdDev).
"""

import sys
import time
import numpy as np
from scipy.io import mmread

def benchmark_scipy_spmv(matrix_path: str, warmup_iters: int = 10, bench_iters: int = 50):
    print(f"Loading matrix: {matrix_path} ...")
    raw_mat = mmread(matrix_path)
    A = raw_mat.tocsr().astype(np.float64)
    
    num_rows, num_cols = A.shape
    nnz = A.nnz
    sparsity = (1.0 - (nnz / (num_rows * num_cols))) * 100.0

    # Match spmv-inspect input vector: all 1.0s
    x = np.ones(num_cols, dtype=np.float64)

    # 10 untimed warmup iterations
    print(f"Executing {warmup_iters} warmup iterations...")
    for _ in range(warmup_iters):
        _ = A.dot(x)

    # 50 timed iterations
    print(f"Executing {bench_iters} timed benchmark iterations...")
    times = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        _ = A.dot(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    median_time = float(np.median(times))
    mean_time = float(np.mean(times))
    stddev_time = float(np.std(times))
    t_min = float(np.min(times))
    t_max = float(np.max(times))

    gflops = (2.0 * nnz) / (median_time * 1e9) if median_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("SciPy (Single-Thread C-Kernel) SpMV Benchmark Report")
    print("=" * 60)
    print(f"Matrix              : {matrix_path}")
    print(f"Dimensions          : {num_rows} x {num_cols}")
    print(f"Non-Zero Elements   : {nnz}")
    print(f"Sparsity            : {sparsity:.4f}%")
    print(f"Protocol            : {warmup_iters} warmup + {bench_iters} timed iterations")
    print("-" * 60)
    print(f"Median Execution Time: {median_time * 1000.0:.4f} ms (± {stddev_time * 1000.0:.4f} ms)")
    print(f"Min / Max Time       : {t_min * 1000.0:.4f} ms / {t_max * 1000.0:.4f} ms")
    print(f"Compute Throughput   : {gflops:.4f} GFLOP/s")
    print("=" * 60)

    return median_time, gflops

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ca-HepPh/ca-HepPh.mtx"
    benchmark_scipy_spmv(path)
