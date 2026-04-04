#!/bin/bash

set -euo pipefail

echo "### Start ci-test-codecov $(date)"

if [ -z "${CODECOV_TOKEN:-}" ]; then
  echo "CODECOV_TOKEN must be defined for Codecov uploads"
  exit 1
fi

WORKDIR="/tmp/proteus-codecov-${CI_JOB_ID}"
ARTIFACT_DIR="${CI_PROJECT_DIR}/gitlab-codecov-artifacts/${CI_MACHINE}"
mkdir -p "${WORKDIR}" "${ARTIFACT_DIR}"
cd "${WORKDIR}"

run_ctest() {
  echo "### $(date) START TESTING ${1} ###"
  shift
  env "$@" ctest -j8 -T test --output-on-failure
  echo "### $(date) END TESTING ###"
}

install_gcovr() {
  if [ "${CI_MACHINE}" = "matrix" ]; then
    PYTHON_BIN=python
    python -m pip install gcovr
  else
    PYTHON_BIN=python3
    python3 -m pip install --user gcovr
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
}

install_codecov_uploader() {
  local codecov_platform
  case "$(uname -m)" in
    x86_64) codecov_platform="linux" ;;
    aarch64|arm64) codecov_platform="linux-arm64" ;;
    *)
      echo "Unsupported architecture for Codecov uploader: $(uname -m)"
      exit 1
      ;;
  esac

  curl -fsSL "https://uploader.codecov.io/latest/${codecov_platform}/codecov" -o codecov
  chmod +x codecov
}

if [ "${CI_MACHINE}" = "matrix" ]; then
  ml load cmake/3.30
  ml load cuda/${PROTEUS_CI_CUDA_VERSION}
  PYTHON_VERSION=3.12

  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"

  MINIFORGE_DIR=miniforge3
  mkdir -p "${MINIFORGE_DIR}"
  wget -q --tries=5 --retry-connrefused --wait=5 \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh \
    -O "./${MINIFORGE_DIR}/miniforge.sh"
  bash "./${MINIFORGE_DIR}/miniforge.sh" -b -u -p "./${MINIFORGE_DIR}"
  rm "./${MINIFORGE_DIR}/miniforge.sh"
  source "./${MINIFORGE_DIR}/bin/activate"
  conda create -y -q -n proteus --override-channels -c conda-forge \
    python=${PYTHON_VERSION} clang=${PROTEUS_CI_LLVM_VERSION} clangxx=${PROTEUS_CI_LLVM_VERSION} \
    clangdev=${PROTEUS_CI_LLVM_VERSION} llvmdev=${PROTEUS_CI_LLVM_VERSION} lit=${PROTEUS_CI_LLVM_VERSION} \
    mlir=${PROTEUS_CI_LLVM_VERSION} gcc=12 gxx=12
  conda activate proteus

  LLVM_INSTALL_DIR=$(llvm-config --prefix)
  CMAKE_OPTIONS_MACHINE=" -DCMAKE_PREFIX_PATH=$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake"
  CMAKE_OPTIONS_MACHINE+=" -DPROTEUS_ENABLE_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90"
  CMAKE_OPTIONS_MACHINE+=" -DCMAKE_CUDA_COMPILER=$LLVM_INSTALL_DIR/bin/clang++"
  CMAKE_OPTIONS_MACHINE+=" -DCMAKE_CUDA_FLAGS=-std=c++17"
  GCOV_EXECUTABLE="$LLVM_INSTALL_DIR/bin/llvm-cov gcov"
else
  ml load rocm/${PROTEUS_CI_ROCM_VERSION}

  if [ -z "${ROCM_PATH:-}" ]; then
    echo "Failed to find ROCM_PATH"
    exit 1
  fi

  echo "=> ROCM_PATH ${ROCM_PATH}"
  LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
  CMAKE_OPTIONS_MACHINE=" -DPROTEUS_ENABLE_HIP=on"
  CMAKE_OPTIONS_MACHINE+=" -DCMAKE_HIP_ARCHITECTURES=gfx942;gfx90a"
  GCOV_EXECUTABLE="$LLVM_INSTALL_DIR/bin/llvm-cov gcov"
fi

install_gcovr
install_codecov_uploader

CMAKE_OPTIONS="-DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR"
CMAKE_OPTIONS+=" -DCMAKE_BUILD_TYPE=Release"
CMAKE_OPTIONS+=" -DCMAKE_C_COMPILER=$LLVM_INSTALL_DIR/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_INSTALL_DIR/bin/clang++"
CMAKE_OPTIONS+=" -DENABLE_DEVELOPER_COMPILER_FLAGS=on"
CMAKE_OPTIONS+=" -DPROTEUS_ENABLE_MLIR=on"
CMAKE_OPTIONS+=" -DENABLE_TESTS=on"
CMAKE_OPTIONS+=" -DENABLE_COVERAGE=on"
CMAKE_OPTIONS+=${CMAKE_OPTIONS_MACHINE}

mkdir build
pushd build

cmake "${CI_PROJECT_DIR}" ${CMAKE_OPTIONS} 2>&1 | tee cmake_output.log
if grep -q "Manually-specified variables were not used by the project:" cmake_output.log; then
  echo "Error: Unused variables detected"
  exit 1
fi

make -j

run_ctest "SYNC COMPILATION KERNEL_CLONE cross-clone" PROTEUS_KERNEL_CLONE=cross-clone
run_ctest "SYNC COMPILATION KERNEL_CLONE link-clone-light" PROTEUS_KERNEL_CLONE=link-clone-light
run_ctest "SYNC COMPILATION KERNEL_CLONE link-clone-prune" PROTEUS_KERNEL_CLONE=link-clone-prune
run_ctest "(BLOCKING) ASYNC COMPILATION" PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1

if [ "${CI_MACHINE}" = "tioga" ] || [ "${CI_MACHINE}" = "tuolumne" ]; then
  run_ctest "SYNC COMPILATION WITH PROTEUS CODEGEN SERIAL" PROTEUS_CODEGEN=serial
  run_ctest "(BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN SERIAL" \
    PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=serial

  if [ "${PROTEUS_CI_ROCM_VERSION}" = "6.4.3" ] \
     || [ "${PROTEUS_CI_ROCM_VERSION}" = "7.1.1" ] \
     || [ "${PROTEUS_CI_ROCM_VERSION}" = "7.2.0" ]; then
    run_ctest "SYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL" PROTEUS_CODEGEN=parallel
    run_ctest "(BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL" \
      PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=parallel
  fi
fi

"${PYTHON_BIN}" -m gcovr \
  --root "${CI_PROJECT_DIR}" \
  --filter "${CI_PROJECT_DIR}/src" \
  --filter "${CI_PROJECT_DIR}/include" \
  --exclude-directories "${CI_PROJECT_DIR}/tests" \
  --gcov-executable "${GCOV_EXECUTABLE}" \
  --xml-pretty \
  --output coverage.xml \
  --print-summary \
  .

cp coverage.xml "${ARTIFACT_DIR}/coverage.xml"

CODECOV_FLAG="gitlab-${CI_MACHINE}"
CODECOV_NAME="gitlab-${CI_MACHINE}-gpu"
./codecov \
  -t "${CODECOV_TOKEN}" \
  -f coverage.xml \
  -F "${CODECOV_FLAG}" \
  -n "${CODECOV_NAME}" \
  -R "${CI_PROJECT_DIR}" \
  -Z 2>&1 | tee "${ARTIFACT_DIR}/codecov-upload.log"

popd
