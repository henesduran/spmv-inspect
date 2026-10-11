#pragma once
#include <vector>
#include <sys/resource.h>

namespace spmv {
struct Metrics {
    double t_max;
    double t_min;
    double t_avg;
    double imb_ratio;

    Metrics(double tMax,double tMin,double tAvg,double ratio): t_max(tMax),t_min(tMin),t_avg(tAvg),imb_ratio(ratio){}
};

inline Metrics compute_metrics(const std::vector<double> &times){
    if (times.empty()){
        return Metrics(0.0,0.0,0.0,0.0);
    }

    double t_avg ,t_min = times[0], t_max = times[0],imb_ratio = 0.0, sum = 0.0;

    for (size_t i = 0 , N = times.size(); i < N; i++)
    {
        sum += times[i];
        if(times[i] < t_min){
            t_min = times[i];
        }
        if(times[i] > t_max){
            t_max = times[i];
        }
    }

    t_avg = sum / times.size();
    if(t_max != 0.0){
        imb_ratio = (t_max - t_avg) / t_max;
    }

    return Metrics(t_max,t_min,t_avg,imb_ratio);    
}

inline double get_peak_memory_mb(){
    struct rusage usage;
    if (getrusage(RUSAGE_SELF,&usage) == 0){
        return usage.ru_maxrss / 1024.0;
    }
    return 0.0;
}

}