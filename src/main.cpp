#include <iostream>
#include <vector>
#include <omp.h>
#include "mmio.hpp"
#include "csr_matrix.hpp"
#include "spmv.hpp"
#include "metrics.hpp"
#include "reporter.hpp"

using namespace spmv;

int main(int argc, char* argv[]) {
    try {
        std::string filename = "samples/tiny.mtx";
        int num_threads = omp_get_max_threads();
        ScheduleType sched = ScheduleType::Static;
        int warmup_iters = 10;
        int bench_iters = 50;

        if (argc > 1) {
            std::string arg1 = argv[1];
            if (arg1 == "--help" || arg1 == "-h") {
                std::cout << "Usage: " << argv[0] << " [matrix_path.mtx] [num_threads] [static|dynamic]\n";
                std::cout << "Default: " << argv[0] << " samples/tiny.mtx <max_threads> static (10 warmup, 50 bench iters)\n";
                return 0;
            }
            filename = arg1;
        }

        if (argc > 2) {
            num_threads = std::stoi(argv[2]);
            if (num_threads <= 0) {
                num_threads = omp_get_max_threads();
            }
        }

        if (argc > 3) {
            sched = (std::string(argv[3]) == "dynamic") ? ScheduleType::Dynamic : ScheduleType::Static;
        }

        auto coo = read_matrix(filename);
        auto csr = coo_to_csr(coo);

        std::vector<double> x(csr.num_cols, 1.0);

        auto bench_result = benchmark_spmv(csr, x, sched, num_threads, 16, warmup_iters, bench_iters);
        auto metrics = compute_benchmark_metrics(bench_result.all_thread_times);
        double peak_mem = get_peak_memory_mb();

        print_benchmark_report(filename, csr, num_threads, sched, metrics, warmup_iters, bench_iters, peak_mem);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}