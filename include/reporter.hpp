#pragma once

#include <iostream>
#include <string>
#include <iomanip> 

#include "csr_matrix.hpp"
#include "spmv.hpp"
#include "metrics.hpp"


inline void print_report(
    const std::string& matrix_name,
    const CSRMatrix& A,
    int num_threads,
    ScheduleType sched,
    const SpMVResult& results,
    const Metrics& metrics,
    double peak_memory_mb
)
{
    double total_elements = static_cast<double>(A.num_rows) * A.num_cols;
    double sparsity = (1.0 - (static_cast<double>(A.nnz) / total_elements)) * 100.0;
    
    double gflops = (metrics.t_max > 0.0) ? (2.0 * A.nnz) / (metrics.t_max * 1e9) : 0.0;
    double bytes_transferred = (A.num_rows + 1) * sizeof(int)
                            + A.nnz * (sizeof(int) + sizeof(double))
                            + A.num_cols * sizeof(double)
                            + A.num_rows * sizeof(double);
    double bandwidth_gb_s = (metrics.t_max > 0.0) ? (bytes_transferred / (metrics.t_max * 1e9)) : 0.0;
    
    int bar_width = 20;
    int num_spaces;

    std::cout << "Matrix: " << matrix_name << std::endl;
    std::cout << "Dimensions: " << A.num_rows << " x " << A.num_cols << std::endl;
    std::cout << "Non-Zero Elements: " << A.nnz << std::endl;
    std::cout << "Sparsity: " << sparsity << "%" << std::endl;

    std::cout << "\nThread Execution Times:" << std::endl;
    for (size_t i = 0; i < num_threads; i++)
    {
        double t = results.thread_times[i];
        int num_blocks = (metrics.t_max > 0.0) ? static_cast<int>((t / metrics.t_max) * bar_width) : 0;
        num_spaces = bar_width - num_blocks;
        std::cout << "Thread " << i << " : [";
        for (int b = 0; b < num_blocks; ++b) std::cout << "#";
        for (int s = 0; s < num_spaces; ++s) std::cout << " ";
        std::cout << "] " << (t * 1000.0) << " ms\n";
    }

    std::cout << "\n[METRICS]" << std::endl;
    std::cout << "  Execution Time (T_max) : " << (metrics.t_max * 1000.0) << " ms" << std::endl;
    std::cout << "  Average Time   (T_avg) : " << (metrics.t_avg * 1000.0) << " ms" << std::endl;
    std::cout << "  Load Imbalance (λ)     : " << metrics.imb_ratio;
    if (metrics.imb_ratio > 0.40) {
        std::cout << "  [WARN: HIGH LOAD IMBALANCE!]";
    }
    std::cout << std::endl;
    std::cout << "  Throughput             : " << gflops << " GFLOP/s" << std::endl;
    std::cout << "  Memory Bandwidth       : " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "  Peak Memory (RSS)      : " << peak_memory_mb << " MB" << std::endl;
}