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
    if (all_thread_times.empty()) {
        return BenchmarkMetrics{};
    }

    size_t num_threads = all_thread_times.size();
    BenchmarkMetrics metrics;
    metrics.thread_stats.resize(num_threads);

    double sum_medians = 0.0;
    double max_median = 0.0;
    double min_median = 1e9;

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

        sum_medians += stats.median;
        if (stats.median > max_median) max_median = stats.median;
        if (stats.median < min_median) min_median = stats.median;
    }

    metrics.t_max = max_median;
    metrics.t_min = (min_median == 1e9) ? 0.0 : min_median;
    metrics.t_avg = sum_medians / num_threads;
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