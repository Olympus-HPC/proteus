#!/bin/bash

set -e

results_dir=results

# NVIDIA Detailed analysis

# ADAM
python vis-scripts/plot-barh-kernel-duration.py \
    --file ${results_dir}/nvidia-cuda-adam-results-profiler.csv -m nvidia -p figure-7
python vis-scripts/plot-barh-metric.py \
    --file ${results_dir}/nvidia-cuda-adam-results-profiler-metrics.csv -m nvidia -c inst_per_warp -p figure-7

# Feynman-Kac
python vis-scripts/plot-barh-metric.py \
    --file ${results_dir}/nvidia-cuda-feynman-kac-results-profiler-metrics.csv -m nvidia -c inst_per_warp -p figure-8
python vis-scripts/plot-barh-kernel-duration.py \
    --file ${results_dir}/nvidia-cuda-feynman-kac-results-profiler.csv -m nvidia -p figure-8

# WSM5
python vis-scripts/plot-barh-metric.py \
    --file ${results_dir}/nvidia-cuda-wsm5-results-profiler-metrics.csv -m nvidia -c inst_per_warp -p figure-9
python vis-scripts/plot-barh-kernel-duration.py \
    --file ${results_dir}/nvidia-cuda-wsm5-results-profiler.csv -m nvidia -p figure-9

# RSBENCH
python vis-scripts/plot-barh-kernel-duration.py \
    --file ${results_dir}/nvidia-cuda-rsbench-results-profiler.csv -m nvidia -p figure-10
python vis-scripts/plot-barh-metric.py \
    --file ${results_dir}/nvidia-cuda-rsbench-results-profiler-metrics.csv -m nvidia -c stall_exec_dependency -p figure-10

#AMD detailed analysis

# ADAM
python vis-scripts/plot-barh-kernel-duration.py -m amd \
    --file ${results_dir}/amd-hip-adam-results-profiler.csv -p figure-7
python vis-scripts/plot-barh-metric.py -m amd  \
    --file ${results_dir}/amd-hip-adam-results-profiler-metrics.csv -c VALUInsts -p figure-7

# Feynman-Kac
python vis-scripts/plot-barh-kernel-duration.py -m amd \
    --file ${results_dir}/amd-hip-feynman-kac-results-profiler.csv -p figure-8
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-feynman-kac-results-profiler-metrics.csv -c VALUInsts -p figure-8

# #WSM5
python vis-scripts/plot-barh-kernel-duration.py -m amd \
    --file ${results_dir}/amd-hip-wsm5-results-profiler.csv -p figure-9
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-wsm5-results-profiler-metrics.csv -c VALUInsts -p figure-9
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-wsm5-results-profiler-metrics.csv -c SALUInsts -p figure-9
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-wsm5-results-profiler-metrics.csv -c MeanOccupancyPerCU -p figure-9
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-wsm5-results-profiler-metrics.csv -c SFetchInsts -p figure-9
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-wsm5-results-profiler-metrics.csv -c VFetchInsts -p figure-9

# RSBENCH
python vis-scripts/plot-barh-kernel-duration.py -m amd \
    --file ${results_dir}/amd-hip-rsbench-results-profiler.csv -p figure-10
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-rsbench-results-profiler-metrics.csv -c VFetchInsts -p figure-10
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-rsbench-results-profiler-metrics.csv -c VALUBusy -p figure-10
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-rsbench-results-profiler-metrics.csv -c  L2CacheHit -p figure-10

# SW4CK
python vis-scripts/plot-barh-kernel-duration.py -m amd \
    --file ${results_dir}/amd-hip-sw4ck-results-profiler.csv -p figure-11
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-sw4ck-results-profiler-metrics.csv -c L2CacheHit -p figure-11
python vis-scripts/plot-barh-metric.py -m amd \
    --file ${results_dir}/amd-hip-sw4ck-results-profiler-metrics.csv -c SALUInsts -p figure-11
