CXX=hipcc

set -e

# Direct runs.
python driver.py -g "hip/*" --compiler ${CXX} -j $PWD/../build-amd -m amd -x direct

# Profiler runs.
python driver.py -g "hip/*" --compiler ${CXX} -j $PWD/../build-amd -m amd -x profiler

# Profiler with metrics runs.
python driver.py -g "hip/adam" --compiler ${CXX} -j $PWD/../build-amd -m amd -x metrics
python driver.py -g "hip/feynman-kac" --compiler ${CXX} -j $PWD/../build-amd -m amd -x metrics
python driver.py -g "hip/wsm5" --compiler ${CXX} -j $PWD/../build-amd -m amd -x metrics
python driver.py -g "hip/rsbench" --compiler ${CXX} -j $PWD/../build-amd -m amd -x metrics
python driver.py -g "hip/sw4ck" --compiler ${CXX} -j $PWD/../build-amd -m amd -x metrics
