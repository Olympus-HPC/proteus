#!/bin/bash
set -e

# Run an MPI program with auto-detected scheduler.
run_mpi() {
    local np=$1
    shift
    if [[ -n "$FLUX_JOB_ID" ]] || command -v flux &>/dev/null; then
        flux run -n$np "$@"
    elif [[ -n "$SLURM_JOB_ID" ]] || command -v srun &>/dev/null; then
        srun -n$np "$@"
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
    -DPROTEUS_ENABLE_HIP=on \
    -DPROTEUS_ENABLE_MPI=on \
    -DENABLE_TESTS=off \
    -DBUILD_SHARED=off

pushd build-proteus
make -j install
popd

# Build the test.
cmake -S ${TEST_DIR} -B build \
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

rm -rf ${CACHE_DIR}/*
echo "=> PASSED mpi-shared-cache (mpi-local-lookup)"

# Run the MPI shared cache test with mpi-remote-lookup backend.
PROTEUS_CACHE_DIR=${CACHE_DIR} \
PROTEUS_OBJECT_CACHE_CHAIN="mpi-remote-lookup" \
run_mpi 4 ./build/mpi_shared_cache

rm -rf ${CACHE_DIR}
echo "=> PASSED mpi-shared-cache (mpi-remote-lookup)"
