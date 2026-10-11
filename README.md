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
4. **Hardware Performance Metrics:** Computes GFLOP/s compute throughput, memory bandwidth (GB/s), and queries Linux kernel physical resident memory (RSS via `getrusage`).

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

---

## Benchmark Results (16 Threads Showdown)

Evaluated on an AMD/Intel 16-thread processor under Linux / WSL2 using matrices from the **SuiteSparse Matrix Collection**:

### 1. Regular Uniform Matrix: `bcsstk17.mtx` (Structural Engineering)
* **Dimensions:** 10,974 x 10,974 | **NNZ:** 428,650 | **Sparsity:** 99.64%

```text
Matrix: data/bcsstk17/bcsstk17.mtx
Threads: 16
Schedule: STATIC
Dimensions: 10974 x 10974
Non-Zero Elements: 428650
Sparsity: 99.6441%

Thread Execution Times:
Thread  0 : [################### ] 0.903 ms
Thread  1 : [################### ] 0.896 ms
...
Thread 14 : [################### ] 0.903 ms
Thread 15 : [##################  ] 0.829 ms

[METRICS]
  Execution Time (T_max) : 0.904 ms
  Average Time   (T_avg) : 0.865 ms
  Load Imbalance (λ)     : 0.0426  (Near-perfect balance!)
  Throughput             : 0.948 GFLOP/s
  Memory Bandwidth       : 5.930 GB/s
  Peak Memory (RSS)      : 25.88 MB
```

---

### 2. Irregular Power-Law Graph: `ca-HepPh.mtx` (ArXiv Collaboration Network)
* **Dimensions:** 12,008 x 12,008 | **NNZ:** 237,010 | **Sparsity:** 99.83%

#### Static Scheduling (`schedule(static)`): Severe Imbalance
Thread 15 was statically assigned dense "hub" author nodes, while other threads idled:
```text
Matrix: data/ca-HepPh/ca-HepPh.mtx
Threads: 16
Schedule: STATIC
Dimensions: 12008 x 12008
Non-Zero Elements: 237010
Sparsity: 99.8356%

Thread Execution Times:
Thread  0 : [                    ] 0.351 ms
Thread  1 : [##########          ] 3.979 ms
...
Thread 15 : [####################] 7.558 ms  <-- 21x slower than Thread 0!

[METRICS]
  Execution Time (T_max) : 7.558 ms
  Average Time   (T_avg) : 3.976 ms
  Load Imbalance (λ)     : 0.4739  [WARN: HIGH LOAD IMBALANCE!]
  Throughput             : 0.062 GFLOP/s
  Memory Bandwidth       : 0.408 GB/s
```

#### Dynamic Scheduling (`schedule(dynamic, 16)`): Balanced Execution
Worker threads dynamically popped 16-row chunks from the task queue, eliminating the bottleneck:
```text
Matrix: data/ca-HepPh/ca-HepPh.mtx
Threads: 16
Schedule: DYNAMIC
Dimensions: 12008 x 12008
Non-Zero Elements: 237010
Sparsity: 99.8356%

Thread Execution Times:
Thread  0 : [####################] 4.335 ms
Thread  1 : [#################   ] 3.864 ms
...
Thread 15 : [##################  ] 3.911 ms  <-- Load evenly distributed!

[METRICS]
  Execution Time (T_max) : 4.335 ms
  Average Time   (T_avg) : 3.949 ms
  Load Imbalance (λ)     : 0.0889  (Imbalance slashed by 81%!)
  Throughput             : 0.109 GFLOP/s  (+74% Increase)
  Memory Bandwidth       : 0.711 GB/s     (+74% Increase)
```

### Head-to-Head Comparison on `ca-HepPh`:

| Metric | Static Scheduling | Dynamic Scheduling | Improvement |
| :--- | :--- | :--- | :--- |
| **Execution Time ($T_{\max}$)** | **7.558 ms** | **4.335 ms** | **42.6% Speedup (1.74x faster)** |
| **Load Imbalance ($\lambda$)** | **0.4739 (Severe)** | **0.0889 (Balanced)** | **81.2% Reduction** |
| **Throughput (GFLOP/s)** | **0.062 GFLOP/s** | **0.109 GFLOP/s** | **+74.3%** |
| **Memory Bandwidth** | **0.408 GB/s** | **0.711 GB/s** | **+74.3%** |

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
