# spmv-inspect

**Sparse Matrix Thread-Level Profiler & Memory Inspector**  
*A modern C++20, OpenMP, and POSIX systems tool for profiling Sparse Matrix-Vector Multiplication (SpMV), cache-line load balancing, and physical memory footprint.*

---

## Overview

Sparse Matrix-Vector Multiplication ($y = A \cdot x$) is one of the most critical kernels in High-Performance Computing (HPC), Graph Analytics, and Scientific Computing. While SpMV on large-scale sparse graphs is typically memory-bandwidth bound at the DRAM level, small-to-medium graphs ($< 16$ MB) reside entirely in CPU cache hierarchy (L2/L3), shifting the primary performance bottleneck to **thread-level load imbalance ($\lambda$)** and loop scheduling overhead.

`spmv-inspect` is an end-to-end inspection and profiling tool that analyzes:
1. **Thread-Level Load Imbalance ($\lambda$):** Quantifies work distribution across CPU cores in sparse graphs.
2. **Scheduling Trade-offs:** Evaluates `static` (row-based), `dynamic` (chunk-based), and `nnz` (workload-aware NNZ-balanced partitioning) strategies.
3. **Synchronization Overhead Isolation (`nowait`):** Separates pure computation time from barrier wait times.
4. **Statistical Benchmarking Rigor:** Employs untimed warmup iterations and reports per-iteration medians and variance ($\pm \sigma$) to filter out OS interrupts.

---

## Architecture & Data Representation

### Matrix Market (`.mtx`) to CSR Transformation
Matrix Market files store coordinates in Coordinate (COO) format. `spmv-inspect` parses both weighted (`real`) and unweighted network (`pattern`) matrices, sorts non-zero entries in $O(N \log N)$, and constructs a **Compressed Sparse Row (CSR)** representation using a single-pass histogram and prefix-sum:

```text
Toy 3x3 Example Matrix                   CSR Format (Zero-Copy Move Semantics)
+-------------------------------+       +-------------------------------------+
| 0  5.0  0                     |       | values   : [5.0, 8.0, 9.0]          |
| 0  0    0                     | ----> | col_idx  : [1,   0,   2  ]          |
| 8.0 0   9.0                   |       | row_ptr  : [0,   1,   1,  3]        |
+-------------------------------+       +-------------------------------------+
```

*For large-scale networks such as `ca-HepPh` (12,008 x 12,008, 99.8% sparsity), CSR compresses the dense memory footprint from ~1.15 GB down to ~2.94 MB, enabling the entire working set to fit within CPU L3 cache.*

* **`values`:** Non-zero values array (size $NNZ$).
* **`col_idx`:** Column index for each non-zero element (size $NNZ$).
* **`row_ptr`:** Index pointer indicating where each row starts and ends in `values` (size $M + 1$).

### Barrier Synchronization & `nowait` Isolation
Standard OpenMP `#pragma omp for` directives introduce an implicit synchronization barrier at the end of the loop construct. Fast worker threads are blocked at this barrier until the slowest thread completes, which severely contaminates per-thread timing metrics with idle waiting time. `spmv-inspect` synchronizes all threads prior to kernel start via `#pragma omp barrier` and appends `nowait` to work-sharing loops:

```text
Without nowait (Contaminated with Barrier Wait Time):
Thread 0 (Fast) : [==== Work ====][..... Idle Barrier Wait .....] --> Measures T_max (Skewed!)
Thread 1 (Slow) : [==================== Work ===================] --> Measures T_max

With nowait & Entry Barrier (Pure Computation Time):
Thread 0 (Fast) : [==== Work ====] --> Measures T_0 (Pure Work!)
Thread 1 (Slow) : [==================== Work ===================] --> Measures T_1 (Bottleneck!)
```

### Workload-Aware Partitioning: NNZ-Balanced Partitioning
Standard row-based static partitioning divides $M$ rows uniformly among $P$ threads ($\approx M/P$ rows/thread). In power-law graphs with skewed degree distributions (e.g., social, citation, and web networks), hub nodes concentrate thousands of non-zero entries into a handful of rows. This causes severe thread load imbalance ($\lambda \gg 0$) under naive static scheduling.

`spmv-inspect` implements **NNZ-Balanced Static Partitioning** by partitioning non-zero entries (`NNZ`) equally rather than rows:
* **Target Non-Zeros per Thread:** `target_nnz = (i * NNZ) / P` for thread partition $i \in [1, P-1]$.
* **Partition Boundary:** Smallest row index $r$ where `row_ptr[r] >= target_nnz`.

In C++, this is cleanly implemented in $O(P \log M)$ preprocessing time using `std::lower_bound`:
```cpp
size_t target_nnz = (static_cast<size_t>(i) * A.nnz) / n_threads;
auto it = std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), static_cast<int>(target_nnz));
bounds[i] = static_cast<int>(it - A.row_ptr.begin());
```

Each thread $t$ directly processes row interval `[bounds[t], bounds[t+1])` without dynamic task queues or runtime lock contention.

> **Boundary Condition:** 1D row-based NNZ partitioning guarantees uniform arithmetic balance provided that the maximum row degree satisfies $d_{\max} \le \frac{NNZ}{P}$. In `ca-HepPh`, $d_{\max} \approx 491 \ll \frac{237,010}{16} \approx 14,813$, ensuring balanced thread chunks without requiring 2D matrix decomposition.

---

## Benchmark Results (16 Threads Evaluation)

Evaluated on **AMD Ryzen 7 8845HS (Zen 4, 8 Cores / 16 Threads, 16 MB L3 Cache)** under Linux / WSL2 using matrices from the **SuiteSparse Matrix Collection**.

> **Methodology:** Every test executes **10 untimed warmup iterations** followed by **50 timed measurement iterations**. Reported figures represent per-thread **Medians and Standard Deviations ($\pm \sigma$)**, filtering out OS scheduling interrupts and cold-cache effects.

### 1. Regular Uniform Matrix: `bcsstk17.mtx` (Structural Engineering)
* **Dimensions:** 10,974 x 10,974 | **NNZ:** 428,650 | **Sparsity:** 99.64% | **Working Set:** 5.11 MB (Fits in 16 MB L3)

```text
Platform: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
Protocol: 10 warmup + 50 timed iterations (reporting Median ± StdDev)
Matrix: data/bcsstk17/bcsstk17.mtx
Threads: 16
Schedule: STATIC
Dimensions: 10974 x 10974
Non-Zero Elements: 428650
Sparsity: 99.6441%
Working Set Size: 5.11 MB

Thread Execution Times (Median ± StdDev):
Thread  0 : [#########           ] 0.0092 ms  (± 0.0028 ms)
Thread  1 : [###############     ] 0.0158 ms  (± 0.0043 ms)
Thread  2 : [#################   ] 0.0178 ms  (± 0.0032 ms)
Thread  3 : [################    ] 0.0167 ms  (± 0.0015 ms)
Thread  4 : [################    ] 0.0165 ms  (± 0.0044 ms)
Thread  5 : [##############      ] 0.0142 ms  (± 0.0191 ms)
Thread  6 : [################    ] 0.0166 ms  (± 0.0145 ms)
Thread  7 : [#################   ] 0.0174 ms  (± 0.0044 ms)
Thread  8 : [################### ] 0.0197 ms  (± 0.0102 ms)
Thread  9 : [####################] 0.0202 ms  (± 0.0083 ms)
Thread 10 : [##################  ] 0.0191 ms  (± 0.0051 ms)
Thread 11 : [##################  ] 0.0185 ms  (± 0.0043 ms)
Thread 12 : [################### ] 0.0195 ms  (± 0.0091 ms)
Thread 13 : [################### ] 0.0194 ms  (± 0.0036 ms)
Thread 14 : [###############     ] 0.0154 ms  (± 0.0027 ms)
Thread 15 : [############        ] 0.0126 ms  (± 0.0049 ms)

[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0202 ms
  Average Time   (T_avg) : 0.0168 ms
  Load Imbalance (λ)     : 0.1694
  Compute Throughput     : 42.4143 GFLOP/s
  Effective Bandwidth    : 265.3449 GB/s (L3 Cache-Resident)
  Peak Memory (RSS)      : 16.00 MB
```

---

### 2. Irregular Power-Law Graph: `ca-HepPh.mtx` (ArXiv Collaboration Network)
* **Dimensions:** 12,008 x 12,008 | **NNZ:** 237,010 | **Sparsity:** 99.83% | **Working Set:** 2.94 MB (Fits in 16 MB L3)

#### Method 1: Naive Row-Static Scheduling (`schedule(static)`)
Equal row counts ($M / P$) assigned regardless of graph topology:
```text
Thread Execution Times (Median ± StdDev):
Thread  0 : [#############       ] 0.0116 ms  (± 0.0020 ms)
Thread  1 : [##################  ] 0.0155 ms  (± 0.0013 ms)
...
Thread  6 : [################### ] 0.0168 ms  (± 0.0215 ms)  <-- Hub node bottleneck
...
Thread 14 : [#############       ] 0.0114 ms  (± 0.0021 ms)  <-- Starving thread
...
[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0169 ms
  Average Time   (T_avg) : 0.0134 ms
  Load Imbalance (λ)     : 0.2041
  Compute Throughput     : 28.1301 GFLOP/s
  Effective Bandwidth    : 183.0327 GB/s (L3 Cache-Resident)
```

#### Method 2: Dynamic Scheduling (`schedule(dynamic, 16)`)
OpenMP dynamic chunk scheduling mitigates static row skew but incurs atomic loop counter coordination overhead:
```text
Thread Execution Times (Median ± StdDev):
Thread  0 : [#################   ] 0.0212 ms  (± 0.0142 ms)
Thread  1 : [################### ] 0.0229 ms  (± 0.0035 ms)
...
Thread 15 : [################### ] 0.0230 ms  (± 0.0042 ms)

[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0237 ms  (Higher latency due to chunk coordination overhead)
  Average Time   (T_avg) : 0.0201 ms
  Load Imbalance (λ)     : 0.1493
  Compute Throughput     : 20.0224 GFLOP/s
  Effective Bandwidth    : 130.2787 GB/s (L3 Cache-Resident)
```

#### Method 3: NNZ-Balanced Static Partitioning (`ScheduleType::NnzBalanced`)
Workload-aware partition based on CSR `row_ptr` prefix-sum binary search:
```text
Thread Execution Times (Median ± StdDev):
Thread  0 : [################### ] 0.0138 ms  (± 0.0018 ms)
Thread  1 : [################### ] 0.0137 ms  (± 0.0001 ms)
...
Thread 10 : [################### ] 0.0141 ms  (± 0.0001 ms)
...
[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0141 ms  (Fastest execution!)
  Average Time   (T_avg) : 0.0137 ms
  Load Imbalance (λ)     : 0.0238  (88.3% Imbalance Reduction vs Static)
  Compute Throughput     : 33.6758 GFLOP/s
  Effective Bandwidth    : 219.1165 GB/s (L3 Cache-Resident)
```

### Comparative Evaluation & Scheduling Trade-offs:

| Metric | 1. Row-Static | 2. Dynamic | 3. NNZ-Balanced (Workload-Aware) | Architectural Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Execution Time ($T_{\max}$)** | 0.0169 ms | 0.0237 ms | **0.0141 ms** | 16.6% lower latency vs Static, 40.5% lower vs Dynamic |
| **Load Imbalance ($\lambda$)** | 0.2041 | 0.1493 | **0.0238** | **88.3% imbalance reduction** |
| **Compute Throughput** | 28.13 GFLOP/s | 20.02 GFLOP/s | **33.68 GFLOP/s** | +5.55 GFLOP/s throughput gain |
| **Effective Bandwidth** | 183.0 GB/s | 130.3 GB/s | **219.1 GB/s** | Cache-resident throughput |
| **Loop Scheduling Cost** | Static ($M/P$) | Atomic chunk counter | Precomputed static partition | Zero runtime lock contention |

#### Latency Analysis:
* **Dynamic Scheduling Overhead at Microsecond Scale:** In sub-millisecond SpMV execution ($< 0.025$ ms), the coordination cost of centralized atomic loop iteration counters causes observable latency increase ($0.0169 \to 0.0237$ ms).
* **NNZ-Balanced Static Partitioning:** Precomputing non-zero boundaries once outside the parallel region allows threads to execute deterministic, balanced row spans without dynamic coordination overhead, mitigating both row-degree topological skew and scheduler overhead.

---

### 3. External Baseline Comparison & Strong Scaling Analysis

To provide a standard reference point, `spmv-inspect` is benchmarked alongside **SciPy's C-implemented CSR SpMV kernel** (`scipy.sparse.csr_matrix.dot`) using the same evaluation protocol (10 warmup + 50 timed iterations, reporting median latency).

#### Benchmark Configuration: `ca-HepPh.mtx` (12,008 x 12,008 | 237,010 NNZ) on AMD Ryzen 7 8845HS:

| Implementation / Engine | Threads | Median Latency ($T$) | Compute Throughput | Notes / Scaling |
| :--- | :---: | :---: | :---: | :--- |
| **SciPy `csr_matrix` (C-Kernel)** | 1 | 0.1470 ms | 3.23 GFLOP/s | Single-threaded reference baseline |
| **`spmv-inspect` (Sequential Baseline)** | 1 | 0.1379 ms | 3.44 GFLOP/s | Single-thread baseline ($1.07\times$ vs SciPy) |
| **`spmv-inspect` (Row-Static)** | 16 | 0.0169 ms | 28.13 GFLOP/s | $8.16\times$ strong scaling ($S_{16}$) |
| **`spmv-inspect` (Dynamic)** | 16 | 0.0237 ms | 20.02 GFLOP/s | $5.82\times$ strong scaling ($S_{16}$) |
| **`spmv-inspect` (NNZ-Balanced)** | 16 | **0.0141 ms** | **33.68 GFLOP/s** | **$9.78\times$ strong scaling ($S_{16}$)** |

#### Key Insights:
1. **Single-Thread Parity:** At $P = 1$, `spmv-inspect` (0.1379 ms) performs on par with SciPy's compiled C kernel (0.1470 ms), noting that at sub-millisecond latencies, Python function call overhead and result array allocation represent measurable factors.
2. **Strong Scaling ($9.78\times$ on 8C/16T):** On an 8-core / 16-thread Zen 4 processor, NNZ-balanced partitioning achieves $9.78\times$ speedup over single-threaded execution ($61.1\%$ parallel efficiency across 16 hardware SMT threads). Beyond SMT throughput, this speedup benefits from aggregate private L2 cache capacity: Zen 4 provides 1 MB private L2 cache per core. While a single thread's 2.94 MB working set exceeds its local L2 (spilling to L3), distributing the matrix across cores allows thread-local data partitions (~0.18 MB) to fit comfortably within private L2 caches. Note that per-thread timers measure pure compute duration inside the OpenMP parallel construct, excluding outer thread fork/join overhead.

---

## Mathematical Metrics

```text
Load Imbalance (λ)  = (T_max - T_avg) / T_max
Compute Throughput  = (2 * NNZ) / (T_max * 1e9)  [GFLOP/s]
Effective Bandwidth = Bytes / (T_max * 1e9)       [GB/s]
  where Bytes       = (M + 1)*4 + NNZ*12 + N*8 + M*8 (Compulsory stream traffic)
Strong Scaling (S)  = T_1 / T_P
Parallel Efficiency = (S / P) * 100%
```

* **`T_max` & `T_avg`:** Computed as the median of per-iteration maximum and average thread runtimes across timed benchmark iterations, measuring pure barrier-to-barrier parallel execution latency.
* **Effective Bandwidth:** Quantifies compulsory algorithmic stream traffic. For cache-resident working sets ($\le 16$ MB L3), this measures on-chip L3 throughput rather than physical DRAM bus saturation.
* **Physical Memory Footprint (RSS):** Queried directly from the Linux kernel using POSIX `getrusage(RUSAGE_SELF, &usage)` via `usage.ru_maxrss`.

---

## Build and Usage

### Prerequisites
* Linux / WSL2 (Ubuntu 20.04+, Debian, Fedora, Arch) or POSIX-compliant environment
* C++20 compliant compiler (GCC 11+ or Clang 13+)
* OpenMP 3.0+
* CMake 3.20+

### Build Instructions
```bash
# Clone and enter the repository
git clone https://github.com/henesduran/spmv-inspect.git
cd spmv-inspect

# Configure and compile with -O3 optimization
cmake -B build
cmake --build build
```

### Running the Profiler
```bash
# Display CLI usage
./build/spmv-inspect --help

# Syntax: ./build/spmv-inspect <path_to_matrix.mtx> [threads] [static|dynamic|nnz]

# Quick verification with included test fixture
./build/spmv-inspect samples/tiny.mtx 4 static

# Run 16-thread benchmark with naive row-static schedule
./build/spmv-inspect data/ca-HepPh/ca-HepPh.mtx 16 static

# Run 16-thread benchmark with OpenMP dynamic schedule
./build/spmv-inspect data/ca-HepPh/ca-HepPh.mtx 16 dynamic

# Run 16-thread benchmark with workload-aware NNZ-balanced partitioning
./build/spmv-inspect data/ca-HepPh/ca-HepPh.mtx 16 nnz

# Run external SciPy C-kernel baseline benchmark
python3 scripts/verify_baseline.py data/ca-HepPh/ca-HepPh.mtx
```

### Downloading Benchmark Datasets (Optional)
To reproduce the real-world SuiteSparse benchmarks locally:
```bash
mkdir -p data && cd data
wget https://suitesparse-collection-website.herokuapp.com/mm/SNAP/ca-HepPh.tar.gz && tar -xzf ca-HepPh.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/mm/HB/bcsstk17.tar.gz && tar -xzf bcsstk17.tar.gz
cd ..
```

---

## Project Structure

```text
spmv-inspect/
├── CMakeLists.txt          # Modern CMake build configuration (-O3, C++20, OpenMP)
├── README.md               # Architecture documentation and benchmark report
├── .gitignore              # Ignores build artifacts and downloaded dataset files
├── include/
│   ├── mmio.hpp            # Matrix Market (.mtx) parser (real & pattern support)
│   ├── csr_matrix.hpp      # CSR data structures with zero-copy move semantics
│   ├── spmv.hpp            # OpenMP SpMV kernel with thread-level timing
│   ├── metrics.hpp         # Load imbalance (λ) & POSIX getrusage memory reader
│   └── reporter.hpp        # ASCII terminal bar-chart visualizer
├── samples/
│   └── tiny.mtx            # 3x3 verification matrix fixture
├── scripts/
│   └── verify_baseline.py  # SciPy C-kernel baseline benchmark script
└── src/
    └── main.cpp            # Command-line interface driver
```

---

## License
Distributed under the MIT License.
