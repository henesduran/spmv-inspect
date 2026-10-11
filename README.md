# spmv-inspect

**Sparse Matrix Thread-Level Profiler & Memory Inspector**  
*A modern C++20, OpenMP, and POSIX systems tool for profiling Sparse Matrix-Vector Multiplication (SpMV), cache-line load balancing, and physical memory footprint.*

---

## Overview

Sparse Matrix-Vector Multiplication ($y = A \cdot x$) is one of the most critical kernels in High-Performance Computing (HPC), Graph Analytics, and Scientific Computing. Unlike dense matrix multiplication, which is arithmetic-intensive and computationally bound, SpMV is strictly **memory-bandwidth bound** and exhibits **irregular data access patterns**.

`spmv-inspect` is an end-to-end performance inspection tool that analyzes:
1. **Thread-Level Load Imbalance ($\lambda$):** Quantifies work distribution across CPU cores in real-world graphs.
2. **Scheduling Trade-offs:** Benchmarks `static` versus `dynamic` OpenMP work-sharing on uniform versus power-law sparse matrices.
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

#### Static Scheduling (`schedule(static)`): Row-Partitioning Skew
Static row-chunk assignment assigns equal row counts regardless of non-zero densities:
```text
Thread Execution Times (Median ± StdDev):
Thread  0 : [##############      ] 0.0121 ms  (± 0.0006 ms)
Thread  1 : [##################  ] 0.0154 ms  (± 0.0007 ms)
...
Thread  6 : [####################] 0.0170 ms  (± 0.0019 ms)  <-- Dense author cluster
...
Thread 14 : [##########          ] 0.0091 ms  (± 0.0063 ms)

[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0170 ms
  Average Time   (T_avg) : 0.0139 ms
  Load Imbalance (λ)     : 0.1868
  Compute Throughput     : 27.8230 GFLOP/s
  Effective Bandwidth    : 181.0345 GB/s (L3 Cache-Resident)
```

#### Dynamic Scheduling (`schedule(dynamic, 16)`): Near-Perfect Balance
Dynamic work-stealing smooths execution times across all 16 workers, reducing load imbalance ($\lambda$) by **87.7%**:
```text
Thread Execution Times (Median ± StdDev):
Thread  0 : [################### ] 0.0231 ms  (± 0.0069 ms)
Thread  1 : [################### ] 0.0233 ms  (± 0.0058 ms)
Thread  2 : [################### ] 0.0230 ms  (± 0.0053 ms)
...
Thread 15 : [################### ] 0.0228 ms  (± 0.0097 ms)  <-- Uniform execution!

[METRICS (Median Across 50 Iterations)]
  Execution Time (T_max) : 0.0234 ms
  Average Time   (T_avg) : 0.0229 ms
  Load Imbalance (λ)     : 0.0229  (Load Imbalance Slashed to ~2%!)
  Compute Throughput     : 20.2625 GFLOP/s
  Effective Bandwidth    : 131.8408 GB/s (L3 Cache-Resident)
```

### Performance Characteristics & Takeaways:

| Metric | Static Scheduling (`ca-HepPh`) | Dynamic Scheduling (`ca-HepPh`) | Impact |
| :--- | :--- | :--- | :--- |
| **Load Imbalance Factor ($\lambda$)** | **0.1868** | **0.0229** | **87.7% Imbalance Reduction** |
| **Thread Variance ($\sigma$)** | Higher variance across threads | Uniform ($\approx 0.023$ ms across all 16 cores) | Ideal load distribution |
| **Working Set Location** | L3 Cache-Resident (2.94 MB < 16 MB L3) | L3 Cache-Resident (2.94 MB < 16 MB L3) | Zero DRAM bottleneck |

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

# Syntax: ./build/spmv-inspect <path_to_matrix.mtx> [threads] [static|dynamic]

# Quick verification with included test fixture
./build/spmv-inspect samples/tiny.mtx 4 static

# Run 16-thread benchmark with static schedule
./build/spmv-inspect data/ca-HepPh/ca-HepPh.mtx 16 static

# Run 16-thread benchmark with dynamic schedule
./build/spmv-inspect data/ca-HepPh/ca-HepPh.mtx 16 dynamic
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
