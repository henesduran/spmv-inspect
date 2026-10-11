#pragma once

#include <vector>
#include <omp.h>
#include "csr_matrix.hpp"

namespace spmv {
enum class ScheduleType {
    Static,
    Dynamic
};

struct SpMVResult {
    std::vector<double> y;
    std::vector<double> thread_times;

    SpMVResult(std::vector<double> y,std::vector<double> thread_times):y(y),thread_times(thread_times){}
};

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
    std::vector<double> thread_times(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int t_id = omp_get_thread_num();
        double t_start = omp_get_wtime();

        if(sched == ScheduleType::Static){
            #pragma omp for schedule(static)
            for (int i = 0; i < A.num_rows; i++)
            {
                double sum = 0.0;
                int row_start = A.row_ptr[i];
                int row_end = A.row_ptr[i+1];
                
                for (int j =row_start ; j < row_end; j++)
                {
                    int col = A.col_idx[j];
                    double val = A.values[j];

                    sum += val * x[col];
                }
                y[i] = sum;
                
            }
        }else{
            #pragma omp for schedule(dynamic,chunk_size)
            for (int i = 0; i < A.num_rows; i++)
            {
                double sum = 0.0;
                int row_start = A.row_ptr[i];
                int row_end = A.row_ptr[i+1];
                
                for (int j =row_start ; j < row_end; j++)
                {
                    int col = A.col_idx[j];
                    double val = A.values[j];

                    sum += val * x[col];
                }
                y[i] = sum;
                
            }
        }

        double t_end = omp_get_wtime();

        thread_times[t_id] = t_end- t_start;
    }

    return SpMVResult(y,thread_times);
}
}