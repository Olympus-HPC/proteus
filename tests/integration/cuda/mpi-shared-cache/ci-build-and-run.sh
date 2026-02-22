#!/bin/bash
set -e

# Run an MPI program with auto-detected scheduler.
run_mpi() {
    local np=$1
    shift
    if [[ -n "$SLURM_JOB_ID" ]] || command -v srun &>/dev/null; then
        # Use --overlap to allow srun to run nested within the CI job's
        # allocated resources.
        # USe --cpu-bind=none to disable CPU binding, which can interfere with
        # the outer srun.
        srun --overlap --cpu-bind=none --gpus=1 -n$np "$@"
    else
        echo "ERROR: No supported MPI launcher found (flux or slurm)"
        exit 1
    fi
}

# Build proteus with MPI enabled.
cmake -S ${CI_PROJECT_DIR} -B build-proteus \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_INSTALL_PREFIX=${PWD}/install-proteus \
    -DPROTEUS_ENABLE_CUDA=on \
    -DPROTEUS_ENABLE_MPI=on \
    -DENABLE_TESTS=off \
    -DBUILD_SHARED=off

pushd build-proteus
make -j install
popd

# Build the test.
cmake -S ${TEST_DIR} -B build \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_CUDA_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -Dproteus_DIR="${PWD}/install-proteus"
pushd build
make -j
popd

# Run the MPI shared cache test with mpi-local-lookup backend.
CACHE_DIR=$(mktemp -d)
echo "Using temporary cache directory: ${CACHE_DIR}"

PROTEUS_CACHE_DIR=${CACHE_DIR} \
PROTEUS_OBJECT_CACHE_CHAIN="mpi-local-lookup" \
run_mpi 4 ./build/mpi_shared_cache

# Hot cache rerun: verify caches work with existing cache files.
PROTEUS_CACHE_DIR=${CACHE_DIR} \
PROTEUS_OBJECT_CACHE_CHAIN="mpi-local-lookup" \
run_mpi 4 ./build/mpi_shared_cache

rm -rf ${CACHE_DIR}/*
echo "=> PASSED mpi-shared-cache (mpi-local-lookup)"

# Single-rank case for mpi-local-lookup.
PROTEUS_CACHE_DIR=${CACHE_DIR} \
PROTEUS_OBJECT_CACHE_CHAIN="mpi-local-lookup" \
run_mpi 1 ./build/mpi_shared_cache

rm -rf ${CACHE_DIR}/*
echo "=> PASSED mpi-shared-cache (mpi-local-lookup, single rank)"

# Run the MPI shared cache test with mpi-remote-lookup backend.
PROTEUS_CACHE_DIR=${CACHE_DIR} \
PROTEUS_OBJECT_CACHE_CHAIN="mpi-remote-lookup" \
run_mpi 4 ./build/mpi_shared_cache

# Hot cache rerun: verify caches work with existing cache files.
PROTEUS_CACHE_DIR=${CACHE_DIR} \
PROTEUS_OBJECT_CACHE_CHAIN="mpi-remote-lookup" \
run_mpi 4 ./build/mpi_shared_cache

rm -rf ${CACHE_DIR}/*
echo "=> PASSED mpi-shared-cache (mpi-remote-lookup)"

# Single-rank corner case for mpi-remote-lookup.
PROTEUS_CACHE_DIR=${CACHE_DIR} \
PROTEUS_OBJECT_CACHE_CHAIN="mpi-remote-lookup" \
run_mpi 1 ./build/mpi_shared_cache

rm -rf ${CACHE_DIR}
echo "=> PASSED mpi-shared-cache (mpi-remote-lookup, single rank)"
