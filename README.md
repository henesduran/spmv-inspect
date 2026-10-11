# spmv-inspect

**Sparse Matrix Thread-Level Profiler & Memory Inspector**  
*A modern C++20, OpenMP, and POSIX systems tool for profiling Sparse Matrix-Vector Multiplication (SpMV), cache-line load balancing, and physical memory footprint.*

---

## Overview

Sparse Matrix-Vector Multiplication ($y = A \cdot x$) is one of the most critical kernels in High-Performance Computing (HPC), Graph Analytics, and Scientific Computing. Unlike dense matrix multiplication, which is arithmetic-intensive and computationally bound, SpMV is strictly **memory-bandwidth bound** and exhibits **irregular data access patterns**.

`spmv-inspect` is an end-to-end performance inspection tool that analyzes:
1. **Thread-Level Load Imbalance ($\lambda$):** Quantifies work distribution across CPU cores in real-world graphs.
2. **Scheduling & Partitioning Trade-offs:** Benchmarks `static` (row-based), `dynamic` (work-stealing), and `nnz` (workload-aware NNZ-balanced static partitioning) on uniform versus power-law sparse matrices.
3. **False Sharing Avoidance:** Employs thread-isolated timing registers to prevent L1/L2 cache-line invalidation storms.
4. **Barrier Wait-Time Isolation (`nowait`):** Eliminates OpenMP worksharing implicit barriers to measure pure CPU compute time rather than idle barrier synchronization wait.
5. **Hardware Performance Metrics:** Computes GFLOP/s compute throughput, memory bandwidth (GB/s), and queries Linux kernel physical resident memory (RSS via `getrusage`).

---

## Architecture & Data Representation

### Matrix Market (`.mtx`) to CSR Transformation
Matrix Market files store coordinates in Coordinate (COO) format. `spmv-inspect` parses both weighted (`real`) and unweighted network (`pattern`) matrices, sorts non-zero entries in $O(N \log N)$, and constructs a **Compressed Sparse Row (CSR)** representation using a single-pass histogram and prefix-sum:

```text
Dense Matrix (12,008 x 12,008)          CSR Format (Zero-Copy Move Semantics)
+-------------------------------+       +-------------------------------------+
| 0  5  0                       |       | values   : [5.0, 8.0, 9.0]          |
| 0  0  0   (>99.8% Zeros!)     | ----> | col_idx  : [1,   0,   2  ]          |
| 8  0  9                       |       | row_ptr  : [0,   1,   1,  3]        |
+-------------------------------+       +-------------------------------------+
   Storage: ~1.15 GB in RAM                 Storage: ~5.3 MB in RAM (99.5% saved!)
```

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
Standard row-based static partitioning divides $M$ rows uniformly among $P$ threads ($\approx \frac{M}{P}$ rows/thread). In power-law graphs with skewed degree distributions (e.g., social, citation, and web networks), hub nodes concentrate thousands of non-zero entries into a handful of rows. This causes severe thread load imbalance ($\lambda \gg 0$) under naive static scheduling.

`spmv-inspect` implements **NNZ-Balanced Static Partitioning** by partitioning the non-zero elements ($NNZ$) equally rather than rows:
$$\text{target\_nnz}_i = \left\lfloor \frac{i \times NNZ}{P} \right\rfloor \quad \text{for } i \in [1, P-1]$$

Since the CSR `row_ptr` array is monotonically non-decreasing, we compute optimal row boundaries using a left binary search (`std::lower_bound`) on `row_ptr` in $O(P \log M)$ precomputation time:
$$\text{bounds}[i] = \text{std::lower\_bound}(\text{row\_ptr.begin}(), \text{row\_ptr.end}(), \text{target\_nnz}_i) - \text{row\_ptr.begin}()$$

Thread $t$ executes its assigned row range $[\text{bounds}[t], \text{bounds}[t+1])$ directly. This achieves:
1. **Zero Runtime Coordination:** Precomputed once before execution; requires zero atomic work queues, locks, or dynamic scheduling overhead.
2. **Deterministic Arithmetic Balancing:** Every thread performs approximately $\frac{NNZ}{P}$ multiply-accumulate operations, eliminating structural tail latency.

---

## Benchmark Results (16 Threads Showdown)

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
  Load Imbalance (λ)     : 0.1694  (Balanced distribution)
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
Thread  0 : [##############      ] 0.0131 ms  (± 0.0017 ms)
Thread  1 : [###############     ] 0.0136 ms  (± 0.0010 ms)
...
Thread  6 : [####################] 0.0175 ms  (± 0.0014 ms)  <-- Hub node bottleneck
...
Thread  9 : [############        ] 0.0107 ms  (± 0.0010 ms)  <-- Starving thread
...
[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0175 ms
  Average Time   (T_avg) : 0.0142 ms
  Load Imbalance (λ)     : 0.1900
  Compute Throughput     : 27.1303 GFLOP/s
  Effective Bandwidth    : 176.5272 GB/s (L3 Cache-Resident)
```

#### Method 2: Dynamic Scheduling (`schedule(dynamic, 16)`)
OpenMP dynamic work-stealing queues eliminate computational imbalance ($\lambda \to 0.0072$) but incur runtime synchronization overhead:
```text
Thread Execution Times (Median ± StdDev):
Thread  0 : [################### ] 0.0190 ms  (± 0.0006 ms)
Thread  1 : [################### ] 0.0193 ms  (± 0.0006 ms)
...
Thread 15 : [####################] 0.0194 ms  (± 0.0019 ms)  <-- Uniform execution

[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0194 ms  (Slower than Static due to queue contention!)
  Average Time   (T_avg) : 0.0193 ms
  Load Imbalance (λ)     : 0.0072
  Compute Throughput     : 24.3888 GFLOP/s
  Effective Bandwidth    : 158.6892 GB/s (L3 Cache-Resident)
```

#### Method 3: NNZ-Balanced Static Partitioning (`ScheduleType::NnzBalanced`)
Workload-aware partition based on CSR `row_ptr` prefix-sum binary search:
```text
Thread Execution Times (Median ± StdDev):
Thread  0 : [################### ] 0.0138 ms  (± 0.0013 ms)
Thread  1 : [################### ] 0.0135 ms  (± 0.0020 ms)
...
Thread 13 : [####################] 0.0142 ms  (± 0.0012 ms)
...
[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0142 ms  (Fastest execution!)
  Average Time   (T_avg) : 0.0138 ms
  Load Imbalance (λ)     : 0.0253  (86.7% Imbalance Reduction vs Static)
  Compute Throughput     : 33.4961 GFLOP/s  (+23.5% vs Static, +37.3% vs Dynamic)
  Effective Bandwidth    : 217.9475 GB/s (L3 Cache-Resident)
```

### Three-Way Performance Showdown & HPC Trade-offs:

| Metric | 1. Row-Static | 2. Dynamic | 3. NNZ-Balanced (Workload-Aware) | Architectural Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Execution Time ($T_{\max}$)** | 0.0175 ms | 0.0194 ms | **0.0142 ms** | **18.9% faster than Static, 26.8% faster than Dynamic** |
| **Load Imbalance ($\lambda$)** | 0.1900 | **0.0072** | **0.0253** | **86.7% Imbalance Reduction without dynamic overhead** |
| **Compute Throughput** | 27.13 GFLOP/s | 24.39 GFLOP/s | **33.50 GFLOP/s** | **Optimal throughput frontier (+6.37 GFLOP/s)** |
| **Effective Bandwidth** | 176.5 GB/s | 158.7 GB/s | **217.9 GB/s** | **Peak L3 bandwidth saturation** |
| **Coordination Overhead** | None | High (Atomic queues) | **Zero (Precomputed prefix-sum)** | **No thread contention on work queues** |

#### Why Dynamic Scheduling Regressed in Latency ($0.0175 \to 0.0194$ ms):
While OpenMP dynamic scheduling achieved near-zero arithmetic imbalance ($\lambda = 0.0072$), total runtime increased by $10.9\%$. In sub-millisecond SpMV kernels ($< 0.02$ ms), the atomic memory contention of work-stealing from the central task pool exceeds the computation time saved by balancing rows.

#### Why NNZ-Balanced Dominated Both:
By mapping the non-zero partition to the monotonically non-decreasing `row_ptr` prefix-sum array, binary search determines non-zero boundaries in $O(P \log M)$ preprocessing time. In the parallel kernel, every worker thread iterates over its designated row span with zero atomic locks and zero branch divergence, simultaneously eliminating both the topological imbalance of static scheduling and the synchronization overhead of dynamic scheduling.

---

## Mathematical Metrics

1. **Load Imbalance Factor ($\lambda$):**
   $$\lambda = \frac{T_{\max} - T_{\text{avg}}}{T_{\max}}$$
   Where $T_{\max} = \max_i(T_i)$ and $T_{\text{avg}} = \frac{1}{P} \sum_{i=0}^{P-1} T_i$.  
   $\lambda = 0.0$ indicates ideal balance; $\lambda \to 1.0$ indicates total thread serialization.

2. **Compute Throughput (GFLOP/s):**
   $$\text{GFLOP/s} = \frac{2 \times NNZ}{T_{\max} \times 10^9}$$

3. **Memory Bandwidth (GB/s):**
   $$\text{Bytes} = (M + 1) \times 4 + NNZ \times (4 + 8) + N \times 8 + M \times 8$$
   $$\text{Bandwidth} = \frac{\text{Bytes}}{T_{\max} \times 10^9}$$

4. **Physical Memory Footprint (RSS):**
   Queried directly from the Linux kernel using `getrusage(RUSAGE_SELF, &usage)` via `usage.ru_maxrss`.

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

# Run 16-thread benchmark with dynamic work-stealing schedule
./build/spmv-inspect data/ca-HepPh/ca-HepPh.mtx 16 dynamic

# Run 16-thread benchmark with workload-aware NNZ-balanced partitioning
./build/spmv-inspect data/ca-HepPh/ca-HepPh.mtx 16 nnz
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
└── src/
    └── main.cpp            # Command-line interface driver
```

---

## License
Distributed under the MIT License.
