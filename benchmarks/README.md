The driver script builds, runs benchmark programs, and collects measurements.
Execute it with `--help` for detailed options.
```
python driver.py --help"
```

The `runscripts` directory contains a script per machine (nvidia, amd) that
invokes the driver to collect all measurements for reproducing the plots/tables
in the manuscript.

The `vis-scripts` directory contains visualization scripts to create plots and
tables after measurements are collected.
Scripts are named after the figure/table they generate in the manuscript.
The `plot-all.sh` script creates all plots/tables invoking individual scripts.

The directory `cuda` contains original (AOT-only) and Proteus implementations of
HeCBench programs, and the directory `cuda-jitify` contains Jitify
implementations of those programs for NVIDIA.
The directory `hip` contains original (AOT-only) and Proteus implementations of
HeCBench programs for AMD.

The directory `external/jitify` contains NVIDIA's Jitify implementation.