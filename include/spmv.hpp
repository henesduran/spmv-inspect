#pragma once

#include <vector>
#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include "csr_matrix.hpp"

namespace spmv {

enum class ScheduleType {
    Static,
    Dynamic,
    NnzBalanced
};

struct SpMVResult {
    std::vector<double> y;
    std::vector<double> thread_times;

    SpMVResult(std::vector<double> y, std::vector<double> thread_times)
        : y(std::move(y)), thread_times(std::move(thread_times)) {}
};

struct BenchmarkResult {
    std::vector<double> y;
    std::vector<std::vector<double>> all_thread_times; // [num_threads][num_iterations]
    int warmup_iters = 0;
    int bench_iters = 0;
};

inline std::vector<int> compute_nnz_partitions(const CSRMatrix &A, int n_threads){
    
    std::vector<int> bounds(n_threads + 1, 0);
    bounds[n_threads] = A.num_rows;

    for (int i = 1; i < n_threads; i++)
    {
        size_t target_nnz = (static_cast<size_t>(i) * A.nnz) / n_threads;
        auto it = std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), static_cast<int>(target_nnz));
        bounds[i] = static_cast<int>(it - A.row_ptr.begin());
    }
    return bounds;
}

inline void spmv_kernel_run(const CSRMatrix &A,
                            const std::vector<double> &x,
                            std::vector<double> &y,
                            ScheduleType sched,
                            int num_threads,
                            int chunk_size,
                            double *thread_times_out,
                            const std::vector<int>& bounds)
{
    #pragma omp parallel num_threads(num_threads)
    {
        int t_id = omp_get_thread_num();
        #pragma omp barrier
        double t_start = omp_get_wtime();

        if (sched == ScheduleType::Static) {
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < A.num_rows; i++) {
                double sum = 0.0;
                int row_start = A.row_ptr[i];
                int row_end = A.row_ptr[i + 1];

                for (int j = row_start; j < row_end; j++) {
                    int col = A.col_idx[j];
                    double val = A.values[j];
                    sum += val * x[col];
                }
                y[i] = sum;
            }
        } else if(sched == ScheduleType::Dynamic){
            #pragma omp for schedule(dynamic, chunk_size) nowait
            for (int i = 0; i < A.num_rows; i++) {
                double sum = 0.0;
                int row_start = A.row_ptr[i];
                int row_end = A.row_ptr[i + 1];

                for (int j = row_start; j < row_end; j++) {
                    int col = A.col_idx[j];
                    double val = A.values[j];
                    sum += val * x[col];
                }
                y[i] = sum;
            }
        }else if(sched==ScheduleType::NnzBalanced){
            int r_start = bounds[t_id];
            int r_end = bounds[t_id+1];

            for(int i = r_start; i < r_end; i++){
                double sum = 0.0;
                int row_start = A.row_ptr[i];
                int row_end   = A.row_ptr[i + 1];

                for (int j = row_start; j < row_end; j++) {
                    sum += A.values[j] * x[A.col_idx[j]];
                }
                y[i] = sum;
            }
        }

        double t_end = omp_get_wtime();
        if (thread_times_out != nullptr) {
            thread_times_out[t_id] = t_end - t_start;
        }
    }
}

inline SpMVResult spmv_csr(const CSRMatrix &A,
                           const std::vector<double> &x,
                           ScheduleType sched,
                           int num_threads,
                           int chunk_size = 16)
{
    if (num_threads <= 0) {
        num_threads = 1;
    }

    if (static_cast<size_t>(A.num_cols) != x.size()) {
        throw std::runtime_error("number of cols in matrix must match vector size!");
    }

    std::vector<double> y(A.num_rows, 0.0);
    std::vector<double> thread_times(num_threads, 0.0);

    std::vector<int> bounds;
    if (sched == ScheduleType::NnzBalanced) {
        bounds = compute_nnz_partitions(A, num_threads);
    }

    spmv_kernel_run(A, x, y, sched, num_threads, chunk_size, thread_times.data(),bounds);

    return SpMVResult(std::move(y), std::move(thread_times));
}

inline BenchmarkResult benchmark_spmv(const CSRMatrix &A,
                                      const std::vector<double> &x,
                                      ScheduleType sched,
                                      int num_threads,
                                      int chunk_size = 16,
                                      int warmup_iters = 10,
                                      int bench_iters = 50)
{
    if (num_threads <= 0) {
        num_threads = 1;
    }

    if (static_cast<size_t>(A.num_cols) != x.size()) {
        throw std::runtime_error("number of cols in matrix must match vector size!");
    }

    std::vector<double> y(A.num_rows, 0.0);

    std::vector<int> bounds;
    if (sched == ScheduleType::NnzBalanced) {
        bounds = compute_nnz_partitions(A, num_threads);
    }

    // Warmup phase (untimed): caches warm, branch predictors train, CPU governor clocks up
    for (int w = 0; w < warmup_iters; ++w) {
        spmv_kernel_run(A, x, y, sched, num_threads, chunk_size, nullptr,bounds);
    }

    // Benchmark phase: record per-thread durations across multiple runs
    std::vector<std::vector<double>> all_thread_times(num_threads, std::vector<double>(bench_iters, 0.0));
    std::vector<double> iter_times(num_threads, 0.0);

    for (int r = 0; r < bench_iters; ++r) {
        spmv_kernel_run(A, x, y, sched, num_threads, chunk_size, iter_times.data(),bounds);
        for (int t = 0; t < num_threads; ++t) {
            all_thread_times[t][r] = iter_times[t];
        }
    }

    return BenchmarkResult{std::move(y), std::move(all_thread_times), warmup_iters, bench_iters};
}

}