#!/bin/bash
set -e

# Run an MPI program with auto-detected scheduler.
run_mpi() {
    local np=$1
    shift
    if command -v flux &>/dev/null; then
        flux run -n$np "$@"
    else
        echo "ERROR: flux MPI launcher not found"
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

# We cannot nest an srun inside this script because it will hang as resources
# are allready occupied by the outer srun. Starting a flux instance allows us to
# use flux run to launch the test without nesting.
if command -v flux &>/dev/null; then
    flux start
    echo "Started flux instance with ID: ${FLUX_INSTANCE_ID}"
else
    echo "flux not found"
fi

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
