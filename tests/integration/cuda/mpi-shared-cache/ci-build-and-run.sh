#!/bin/bash
set -e

# Run an MPI program with auto-detected scheduler.
run_mpi() {
    local np=$1
    shift
    if [[ -n "$SLURM_JOB_ID" ]] || command -v srun &>/dev/null; then
        # Nested srun within the CI job's outer srun allocation:
        #   --overlap    : allow nested step to share the parent step's resources
        #   --cpu-bind=none : disable CPU binding to avoid conflicts with the
        #                     outer step's CPU cgroup constraints
        #   env CUDA_VISIBLE_DEVICES=0 : force GPU visibility after srun step
        #                     setup, because Slurm's GPU binding logic re-derives
        #                     CUDA_VISIBLE_DEVICES from the job-level allocation
        #                     during step initialization, overwriting any value
        #                     set via --export
        srun --overlap --cpu-bind=none -n$np env CUDA_VISIBLE_DEVICES=0 "$@"

    else
        echo "ERROR: No supported MPI launcher found (flux or slurm)"
        exit 1
    fi
}

# Build proteus with MPI enabled. Conda's linker does not follow the default
# Matrix MVAPICH transitive library search paths during CMake's MPI link probe,
# so make those paths explicit without changing the MPI selected by the module.
MPI_RPATH_LINK_DIRS=()
if [[ -n "${CONDA_PREFIX}" ]]; then
    MPI_RPATH_LINK_DIRS+=(
        "${CONDA_PREFIX}/lib"
        "${CONDA_PREFIX}/x86_64-conda-linux-gnu/lib"
    )
fi
MPI_RPATH_LINK_DIRS+=(
    /lib64
    /usr/tce/backend/installations/linux-rhel8-x86_64/gcc-10.3.1/intel-oneapi-compilers-2022.1.0-43xp3r52jx2q2rkf3ctzvskqu572xbky/compiler/2022.1.0/linux/compiler/lib/intel64_lin
    /usr/tce/backend/installations/linux-rhel8-x86_64/intel-2021.6.0/mvapich2-2.3.7-2575ifqlr5fbj34wdlj2fo2tmqdrehia/lib
    /usr/tce/backend/installations/linux-rhel8-x86_64/gcc-8.5.0/zlib-1.2.13-pwvjwcyuqiscffdc2x2lr7ti355xwohq/lib
)
MPI_RPATH_LINK_FLAGS=""
for dir in "${MPI_RPATH_LINK_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        MPI_RPATH_LINK_FLAGS+="${MPI_RPATH_LINK_FLAGS:+ }-Wl,-rpath-link,${dir}"
    fi
done
MPI_CMAKE_ARGS=(
    -DCMAKE_EXE_LINKER_FLAGS="${MPI_RPATH_LINK_FLAGS}"
)

cmake -S ${CI_PROJECT_DIR} -B build-proteus \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_INSTALL_PREFIX=${PWD}/install-proteus \
    -DPROTEUS_ENABLE_CUDA=on \
    -DPROTEUS_ENABLE_MPI=on \
    -DENABLE_TESTS=off \
    -DBUILD_SHARED=off \
    "${MPI_CMAKE_ARGS[@]}"

pushd build-proteus
make -j install
popd

# Build the test.
cmake -S ${TEST_DIR} -B build \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_CUDA_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -Dproteus_DIR="${PWD}/install-proteus" \
    "${MPI_CMAKE_ARGS[@]}"
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
