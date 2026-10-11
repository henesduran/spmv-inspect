#include <iostream>
#include <vector>
#include <omp.h>
#include "mmio.hpp"
#include "csr_matrix.hpp"
#include "spmv.hpp"
#include "metrics.hpp"
#include "reporter.hpp"

using namespace spmv;

int main(int argc,char* argv[]) {
    std::string filename = "samples/tiny.mtx";
    int num_threads = omp_get_max_threads();
    ScheduleType sched = ScheduleType::Static;

    if (argc > 1){
        filename = argv[1];
    }

    if (argc > 2){
        num_threads = std::stoi(argv[2]);
    }

    if (argc > 3){
        sched = (std::string(argv[3]) == "dynamic") ? ScheduleType::Dynamic : ScheduleType::Static;
    }

    auto coo = read_matrix(filename);
    auto csr = coo_to_csr(coo);

    std::vector<double> x(csr.num_cols, 1.0);

    
    

    auto result = spmv_csr(csr, x, sched, num_threads);

    auto metrics = compute_metrics(result.thread_times);

    double peak_mem = get_peak_memory_mb();

    print_report(filename, csr, num_threads, sched, result, metrics, peak_mem);

    return 0;
}