#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <sys/resource.h>

namespace spmv {

struct ThreadStats {
    double median = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
};

struct BenchmarkMetrics {
    double t_max = 0.0;
    double t_min = 0.0;
    double t_avg = 0.0;
    double imb_ratio = 0.0;
    std::vector<ThreadStats> thread_stats;
};

inline double calculate_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) {
        return (v[n / 2 - 1] + v[n / 2]) / 2.0;
    }
    return v[n / 2];
}

inline BenchmarkMetrics compute_benchmark_metrics(const std::vector<std::vector<double>> &all_thread_times) {
    if (all_thread_times.empty() || all_thread_times[0].empty()) {
        return BenchmarkMetrics{};
    }

    size_t num_threads = all_thread_times.size();
    size_t num_iters = all_thread_times[0].size();
    BenchmarkMetrics metrics;
    metrics.thread_stats.resize(num_threads);

    // 1. Per-thread statistics for terminal visualization bars
    for (size_t t = 0; t < num_threads; ++t) {
        const auto &samples = all_thread_times[t];
        if (samples.empty()) continue;

        ThreadStats stats;
        stats.median = calculate_median(samples);
        stats.min_val = *std::min_element(samples.begin(), samples.end());
        stats.max_val = *std::max_element(samples.begin(), samples.end());

        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        stats.mean = sum / samples.size();

        double var_sum = 0.0;
        for (double val : samples) {
            double diff = val - stats.mean;
            var_sum += diff * diff;
        }
        stats.stddev = std::sqrt(var_sum / samples.size());

        metrics.thread_stats[t] = stats;
    }

    // 2. Kernel execution duration: in each iteration r, the barrier-to-barrier runtime
    // is dictated by the slowest thread in that iteration.
    std::vector<double> iter_maxes(num_iters, 0.0);
    std::vector<double> iter_avgs(num_iters, 0.0);

    for (size_t r = 0; r < num_iters; ++r) {
        double r_max = 0.0;
        double r_sum = 0.0;
        for (size_t t = 0; t < num_threads; ++t) {
            double val = all_thread_times[t][r];
            if (val > r_max) r_max = val;
            r_sum += val;
        }
        iter_maxes[r] = r_max;
        iter_avgs[r] = r_sum / num_threads;
    }

    metrics.t_max = calculate_median(iter_maxes);
    metrics.t_avg = calculate_median(iter_avgs);
    metrics.t_min = *std::min_element(iter_maxes.begin(), iter_maxes.end());

    if (metrics.t_max > 0.0) {
        metrics.imb_ratio = (metrics.t_max - metrics.t_avg) / metrics.t_max;
    }

    return metrics;
}

inline double get_peak_memory_mb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss / 1024.0;
    }
    return 0.0;
}

}