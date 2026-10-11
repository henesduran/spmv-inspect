#include <iostream>
#include <vector>
#include <omp.h>
#include "mmio.hpp"
#include "csr_matrix.hpp"
#include "spmv.hpp"
#include "metrics.hpp"
#include "reporter.hpp"

int main() {
    std::string filename = "samples/tiny.mtx";
    auto coo = read_matrix(filename);
    auto csr = coo_to_csr(coo);

    std::vector<double> x(csr.num_cols, 1.0);

    int num_threads = 4;
    ScheduleType sched = ScheduleType::Static;

    auto result = spmv_csr(csr, x, sched, num_threads);

    auto metrics = compute_metrics(result.thread_times);

    double peak_mem = get_peak_memory_mb();

    print_report(filename, csr, num_threads, sched, result, metrics, peak_mem);

    return 0;
}