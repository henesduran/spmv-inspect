#pragma once

#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>

#include "csr_matrix.hpp"
#include "spmv.hpp"
#include "metrics.hpp"

namespace spmv {

inline std::string get_cpu_model() {
    std::ifstream file("/proc/cpuinfo");
    if (!file.is_open()) return "POSIX/x86_64 Processor";
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("model name", 0) == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string model = line.substr(colon + 1);
                size_t start = model.find_first_not_of(" \t");
                if (start != std::string::npos) return model.substr(start);
            }
        }
    }
    return "POSIX/x86_64 Processor";
}

inline void print_benchmark_report(
    const std::string& matrix_name,
    const CSRMatrix& A,
    int num_threads,
    ScheduleType sched,
    const BenchmarkMetrics& metrics,
    int warmup_iters,
    int bench_iters,
    double peak_memory_mb
)
{
    double total_elements = static_cast<double>(A.num_rows) * A.num_cols;
    double sparsity = (total_elements > 0.0) ? (1.0 - (static_cast<double>(A.nnz) / total_elements)) * 100.0 : 0.0;
    
    double gflops = (metrics.t_max > 0.0) ? (2.0 * A.nnz) / (metrics.t_max * 1e9) : 0.0;
    double bytes_transferred = (A.num_rows + 1) * sizeof(int)
                            + A.nnz * (sizeof(int) + sizeof(double))
                            + A.num_cols * sizeof(double)
                            + A.num_rows * sizeof(double);
    double bandwidth_gb_s = (metrics.t_max > 0.0) ? (bytes_transferred / (metrics.t_max * 1e9)) : 0.0;
    double working_set_mb = bytes_transferred / (1024.0 * 1024.0);

    int bar_width = 20;

    std::cout << "Platform: " << get_cpu_model() << "\n";
    std::cout << "Protocol: " << warmup_iters << " warmup + " << bench_iters << " timed iterations (reporting Median ± StdDev)\n";
    std::cout << "Matrix: " << matrix_name << "\n";
    std::cout << "Threads: " << num_threads << "\n";

    std::string sched_str = "STATIC";
    if (sched == ScheduleType::Dynamic) sched_str = "DYNAMIC";
    else if (sched == ScheduleType::NnzBalanced) sched_str = "NNZ-BALANCED";
    
    std::cout << "Schedule: " << sched_str << "\n";
    std::cout << "Dimensions: " << A.num_rows << " x " << A.num_cols << "\n";
    std::cout << "Non-Zero Elements: " << A.nnz << "\n";
    std::cout << "Sparsity: " << sparsity << "%\n";
    std::cout << "Working Set Size: " << std::fixed << std::setprecision(2) << working_set_mb << " MB\n";

    std::cout << "\nThread Execution Times (Median ± StdDev):\n";
    for (int i = 0; i < num_threads; ++i)
    {
        const auto& st = metrics.thread_stats[i];
        double t = st.median;
        int num_blocks = (metrics.t_max > 0.0) ? static_cast<int>((t / metrics.t_max) * bar_width) : 0;
        int num_spaces = bar_width - num_blocks;
        std::cout << "Thread " << std::setw(2) << i << " : [";
        for (int b = 0; b < num_blocks; ++b) std::cout << "#";
        for (int s = 0; s < num_spaces; ++s) std::cout << " ";
        std::cout << "] " << std::fixed << std::setprecision(4) << (t * 1000.0) << " ms  (± " << (st.stddev * 1000.0) << " ms)\n";
    }

    std::cout << "\n[METRICS (Median Across " << bench_iters << " Iterations)]" << "\n";
    std::cout << "  Execution Time (T_max) : " << (metrics.t_max * 1000.0) << " ms\n";
    std::cout << "  Average Time   (T_avg) : " << (metrics.t_avg * 1000.0) << " ms\n";
    std::cout << "  Load Imbalance (λ)     : " << metrics.imb_ratio;
    if (metrics.imb_ratio > 0.40) {
        std::cout << "  [WARN: HIGH LOAD IMBALANCE!]";
    }
    std::cout << "\n";
    std::cout << "  Compute Throughput     : " << gflops << " GFLOP/s\n";
    std::cout << "  Effective Bandwidth    : " << bandwidth_gb_s << " GB/s ("
              << (working_set_mb <= 16.0 ? "L3 Cache-Resident" : "DRAM-Bound") << ")\n";
    std::cout << "  Peak Memory (RSS)      : " << peak_memory_mb << " MB\n";
}

}