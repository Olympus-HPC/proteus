#!/bin/bash

set -e

# Figure 3
python vis-scripts/figure-3-plot-bar-end2end-speedup.py --dir results --plot-dir plots -m nvidia
python vis-scripts/figure-3-plot-bar-end2end-speedup.py --dir results --plot-dir plots -m amd

# Table 2
python vis-scripts/table-2-generate.py --dir results --outdir plots

# Table 3
python vis-scripts/table-3-generate.py --dir results --outdir plots

# Figure 4
python vis-scripts/figure-4-plot-bar-kernel-speedup.py  --dir results --plot-dir plots -m nvidia

# Figure 5
python vis-scripts/figure-5-plot-bar-compilation-slowdown.py --dir results --plot-dir plots -m nvidia
python vis-scripts/figure-5-plot-bar-compilation-slowdown.py --dir results --plot-dir plots -m amd

# Figure 6
python vis-scripts/figure-6-plot-bar-end2end-speedup-noopt.py --dir results --plot-dir plots -m nvidia
python vis-scripts/figure-6-plot-bar-end2end-speedup-noopt.py --dir results --plot-dir plots -m amd

# Figures 7 to 11
bash vis-scripts/figures-7to11-plot.sh
