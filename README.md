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
Standard row-based static partitioning divides $M$ rows uniformly among $P$ threads ($\approx \frac{M}{P}$ rows/thread). In power-law graphs with skewed degree distributions (e.g., social, citation, and web networks), hub nodes concentrate thousands of non-zero entries into a handful of rows. This causes severe thread load imbalance ($\lambda \gg 0$) under naive static scheduling.

`spmv-inspect` implements **NNZ-Balanced Static Partitioning** by partitioning non-zero entries ($NNZ$) equally rather than rows:
$$\tau_i = \left\lfloor \frac{i \cdot NNZ}{P} \right\rfloor \quad \text{for } i \in [1, P-1]$$

Because the CSR `row_ptr` prefix-sum array is monotonically non-decreasing, partition boundaries are determined via binary search in $O(P \log M)$ preprocessing time:
$$\text{bounds}[i] = \min \{ r \in [0, M] \mid \text{row\_ptr}[r] \ge \tau_i \}$$

In C++, this is cleanly implemented using `std::lower_bound`:
```cpp
size_t target_nnz = (static_cast<size_t>(i) * A.nnz) / n_threads;
auto it = std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), static_cast<int>(target_nnz));
bounds[i] = static_cast<int>(it - A.row_ptr.begin());
```

Each thread $t$ directly processes row interval $[\text{bounds}[t], \text{bounds}[t+1])$ without dynamic task queues or runtime lock contention.

> **Boundary Condition:** 1D row-based NNZ partitioning guarantees uniform arithmetic balance provided that the maximum row degree satisfies $d_{\max} \le \frac{NNZ}{P}$. In `ca-HepPh`, $d_{\max} \approx 491 \ll \frac{237,010}{16} \approx 14,813$, ensuring balanced thread chunks without requiring 2D matrix decomposition.

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

### Comparative Evaluation & Scheduling Trade-offs:

| Metric | 1. Row-Static | 2. Dynamic | 3. NNZ-Balanced (Workload-Aware) | Architectural Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Execution Time ($T_{\max}$)** | 0.0175 ms | 0.0194 ms | **0.0142 ms** | 18.9% lower latency vs Static |
| **Load Imbalance ($\lambda$)** | 0.1900 | **0.0072** | **0.0253** | 86.7% imbalance reduction |
| **Compute Throughput** | 27.13 GFLOP/s | 24.39 GFLOP/s | **33.50 GFLOP/s** | +6.37 GFLOP/s throughput gain |
| **Effective Bandwidth** | 176.5 GB/s | 158.7 GB/s | **217.9 GB/s** | Cache-resident throughput |
| **Loop Scheduling Cost** | Static ($M/P$) | Atomic chunk counter | Precomputed static partition | Zero runtime lock contention |

#### Latency Analysis:
* **Dynamic Scheduling Overhead at Microsecond Scale:** While OpenMP dynamic scheduling achieved near-zero arithmetic imbalance ($\lambda = 0.0072$), total runtime increased by $10.9\%$ ($0.0175 \to 0.0194$ ms). At sub-millisecond execution times, the coordination cost of centralized atomic loop iteration counters can outweigh the work-balancing gains.
* **NNZ-Balanced Static Partitioning:** By precomputing non-zero boundaries once outside the parallel region, threads execute deterministic, balanced row spans without dynamic coordination overhead, mitigating both row-degree topological skew and scheduler overhead.

---

### 3. Baseline Verification & Strong Scaling Analysis

To provide a standard reference point, `spmv-inspect` is benchmarked alongside **SciPy's C-implemented CSR SpMV kernel** (`scipy.sparse.csr_matrix.dot`) using the same evaluation protocol (10 warmup + 50 timed iterations, reporting median latency).

#### Benchmark Configuration: `ca-HepPh.mtx` (12,008 x 12,008 | 237,010 NNZ) on AMD Ryzen 7 8845HS:

| Implementation / Engine | Threads | Median Latency ($T$) | Compute Throughput | Notes / Scaling |
| :--- | :---: | :---: | :---: | :--- |
| **SciPy `csr_matrix` (C-Kernel)** | 1 | 0.1470 ms | 3.23 GFLOP/s | Single-threaded reference baseline |
| **`spmv-inspect` (Sequential Baseline)** | 1 | 0.1371 ms | 3.46 GFLOP/s | Single-thread baseline ($1.07\times$ vs SciPy) |
| **`spmv-inspect` (Row-Static)** | 16 | 0.0175 ms | 27.13 GFLOP/s | $7.83\times$ strong scaling ($S_{16}$) |
| **`spmv-inspect` (Dynamic)** | 16 | 0.0194 ms | 24.39 GFLOP/s | $7.07\times$ strong scaling ($S_{16}$) |
| **`spmv-inspect` (NNZ-Balanced)** | 16 | **0.0142 ms** | **33.50 GFLOP/s** | **$9.65\times$ strong scaling ($S_{16}$)** |

#### Key Insights:
1. **Single-Thread Parity:** At $P = 1$, `spmv-inspect` (0.1371 ms) performs on par with SciPy's compiled C kernel (0.1470 ms), noting that at sub-millisecond latencies, Python function call overhead and result array allocation represent measurable factors.
2. **Strong Scaling ($9.65\times$ on 8C/16T):** On an 8-core / 16-thread Zen 4 processor, NNZ-balanced partitioning achieves a strong scaling speedup of **$9.65\times$** over single-threaded execution ($60.3\%$ parallel efficiency across 16 hardware SMT threads).

---

## Mathematical Metrics

1. **Load Imbalance Factor ($\lambda$):**
   $$\lambda = \frac{T_{\max} - T_{\text{avg}}}{T_{\max}}$$
   Where $T_{\max} = \text{median}(\text{iter\_max})$ and $T_{\text{avg}} = \text{median}(\text{iter\_avg})$ across timed iterations, measuring barrier-to-barrier execution latency. $\lambda = 0.0$ indicates ideal load balance; $\lambda \to 1.0$ indicates severe thread serialization.

2. **Compute Throughput (GFLOP/s):**
   $$\text{GFLOP/s} = \frac{2 \times NNZ}{T_{\max} \times 10^9}$$

3. **Effective Memory Bandwidth (GB/s):**
   $$\text{Bytes} = (M + 1) \times 4 + NNZ \times (4 + 8) + N \times 8 + M \times 8$$
   $$\text{Bandwidth} = \frac{\text{Bytes}}{T_{\max} \times 10^9}$$
   *Note: Represents compulsory algorithmic stream traffic. For cache-resident working sets ($\le 16$ MB L3), this quantifies on-chip L3 cache throughput rather than physical DRAM saturation.*

4. **Physical Memory Footprint (RSS):**
   Queried directly from the Linux kernel using `getrusage(RUSAGE_SELF, &usage)` via `usage.ru_maxrss`.

5. **Strong Scaling Speedup & Parallel Efficiency:**
   $$S_P = \frac{T_1}{T_P}, \quad E_P = \frac{S_P}{P} \times 100\%$$
   Where $T_1$ is single-threaded runtime and $P$ is hardware thread count.

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

# Run external SciPy C-kernel baseline benchmark & verification
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
│   └── verify_baseline.py  # SciPy C-kernel baseline & verification script
└── src/
    └── main.cpp            # Command-line interface driver
```

---

## License
Distributed under the MIT License.
