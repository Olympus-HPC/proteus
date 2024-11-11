CXX=${CONDA_PREFIX}/bin/clang++

set -e

# Direct runs.
python driver.py -g "cuda/*" --compiler ${CXX} -j $PWD/../build-nvidia -m nvidia -x direct
python driver.py -g "cuda-jitify/*" --compiler ${CXX} -j $PWD/../build-nvidia -m nvidia -x direct

# Profiler runs.
python driver.py -g "cuda/*" --compiler ${CXX} -j $PWD/../build-nvidia -m nvidia -x profiler
python driver.py -g "cuda-jitify/*" --compiler ${CXX} -j $PWD/../build-nvidia -m nvidia -x profiler

# Profiler with metrics runs.
python driver.py -g "cuda/adam" --compiler ${CXX} -j $PWD/../build-nvidia -m nvidia -x metrics
python driver.py -g "cuda/feynman-kac" --compiler ${CXX} -j $PWD/../build-nvidia -m nvidia -x metrics
python driver.py -g "cuda/wsm5" --compiler ${CXX} -j $PWD/../build-nvidia -m nvidia -x metrics
python driver.py -g "cuda/rsbench" --compiler ${CXX} -j $PWD/../build-nvidia -m nvidia -x metrics
