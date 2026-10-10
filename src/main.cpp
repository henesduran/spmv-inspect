#include <iostream>
#include <vector>
#include <omp.h>
#include "mmio.hpp"
#include "csr_matrix.hpp"
#include "spmv.hpp"

int main() {
    auto coo = read_matrix("samples/tiny.mtx");
    auto csr = coo_to_csr(coo);

    std::vector<double> x(csr.num_cols, 1.0);

    int num_threads = 4;

    std::cout << "\n--- Static Schedule ---" << std::endl;
    auto result1 = spmv_csr(csr, x, ScheduleType::Static, num_threads);

    std::cout << "--- SPMV RESULT (y = A * x) ---" << std::endl;
    for (size_t i = 0; i < result1.y.size(); ++i) {
        std::cout << "y[" << i << "] = " << result1.y[i] << std::endl;
    }

    std::cout << "\n--- THREAD TIMES ---" << std::endl;
    for (int t = 0; t < num_threads; ++t) {
        std::cout << "Thread " << t << " Time: " << result1.thread_times[t] << " seconds\n";
    }



    std::cout << "\n--- Dynamic Schedule ---" << std::endl;

    auto result2 = spmv_csr(csr, x, ScheduleType::Dynamic, num_threads);

    std::cout << "--- SPMV RESULT (y = A * x) ---" << std::endl;
    for (size_t i = 0; i < result2.y.size(); ++i) {
        std::cout << "y[" << i << "] = " << result2.y[i] << std::endl;
    }

    std::cout << "\n--- THREAD TIMES ---" << std::endl;
    for (int t = 0; t < num_threads; ++t) {
        std::cout << "Thread " << t << " Time: " << result2.thread_times[t] << " seconds\n";
    }

    return 0;
}