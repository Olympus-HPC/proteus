#!/bin/bash

set -e

mkdir -p /tmp/proteus-ci-${CI_JOB_ID}
cd /tmp/proteus-ci-${CI_JOB_ID}

if [ "${CI_MACHINE}" == "lassen" ]; then
  ml load cmake/3.23.1
  ml load cuda/12.2.2

  # Install Clang/LLVM through conda.
  MINICONDA_DIR=miniconda3
  mkdir -p ${MINICONDA_DIR}
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ./${MINICONDA_DIR}/miniconda.sh
  bash ./${MINICONDA_DIR}/miniconda.sh -b -u -p ./${MINICONDA_DIR}
  rm ./${MINICONDA_DIR}/miniconda.sh
  source ./${MINICONDA_DIR}/bin/activate
  conda create -y -n proteus -c conda-forge \
      python=3.10 clang=17.0.5 clangxx=17.0.5 llvmdev=17.0.5 lit=17.0.5
  conda activate proteus

  LLVM_INSTALL_DIR=$(llvm-config --prefix)
  CMAKE_OPTIONS_MACHINE=" -DCMAKE_PREFIX_PATH=$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake"
  CMAKE_OPTIONS_MACHINE+=" -DPROTEUS_LINK_SHARED_LLVM=on"
  CMAKE_OPTIONS_MACHINE+=" -DPROTEUS_ENABLE_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70"
  CMAKE_OPTIONS_MACHINE+=" -DCMAKE_CUDA_COMPILER=$LLVM_INSTALL_DIR/bin/clang++"
elif [ "${CI_MACHINE}" == "tioga" ]; then
  ml load rocm/${PROTEUS_CI_ROCM_VERSION}

  if [ -n "$ROCM_PATH" ]; then
    echo "=> ROCM_PATH ${ROCM_PATH}"
  else
    echo "=> Failed to find ROCM_PATH"
    exit 1
  fi

  LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
  CMAKE_OPTIONS_MACHINE=" -DPROTEUS_ENABLE_HIP=on"
  CMAKE_OPTIONS_MACHINE+=" -DCMAKE_HIP_ARCHITECTURES=gfx942;gfx90a"
else
  echo "Unsupported machine ${CI_MACHINE}"
  exit 1
fi

CMAKE_OPTIONS="-DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR"
CMAKE_OPTIONS+=" -DCMAKE_BUILD_TYPE=Release"
CMAKE_OPTIONS+=" -DCMAKE_C_COMPILER=$LLVM_INSTALL_DIR/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_INSTALL_DIR/bin/clang++"
CMAKE_OPTIONS+=${CMAKE_OPTIONS_MACHINE}
CMAKE_OPTIONS+=" -DPROTEUS_ENABLE_DEBUG=${PROTEUS_CI_ENABLE_DEBUG} -DPROTEUS_ENABLE_TIME_TRACING=${PROTEUS_CI_ENABLE_TIME_TRACING}"
if [ -n "${PROTEUS_CI_BUILD_SHARED}" ]; then
  CMAKE_OPTIONS+=" -DBUILD_SHARED=${PROTEUS_CI_BUILD_SHARED}"
fi

mkdir build
pushd build

cmake ${CI_PROJECT_DIR} ${CMAKE_OPTIONS} |& tee cmake_output.log
if grep -q "Manually-specified variables were not used by the project:" cmake_output.log; then
    echo "Error: Unused variables detected"
    exit 1
fi
make -j

if  [ "$PROTEUS_CI_ENABLE_DEBUG" == "on" ] || [ "$PROTEUS_CI_ENABLE_TIME_TRACING" == "on" ]; then
  echo "Skipping tests for debug output or time tracing builds."
  exit 0
fi

# Test synchronous compilation (default) and kernel clone options.
echo "### TESTING SYNC COMPILATION KERNEL_CLONE cross-clone ###"
PROTEUS_KERNEL_CLONE=cross-clone ctest -T test --output-on-failure
echo "### END TESTING SYNC COMPILATION KERNEL_CLONE cross-clone ###"

echo "### TESTING SYNC COMPILATION KERNEL_CLONE link-clone-light ###"
PROTEUS_KERNEL_CLONE=link-clone-light ctest -T test --output-on-failure
echo "### END TESTING SYNC COMPILATION KERNEL_CLONE link-clone-light ###"

echo "### TESTING SYNC COMPILATION KERNEL_CLONE link-clone-prune ###"
PROTEUS_KERNEL_CLONE=link-clone-prune ctest -T test --output-on-failure
echo "### END TESTING SYNC COMPILATION KERNEL_CLONE link-clone-prune ###"

# Test asynchronous compilation.
echo "### TESTING (BLOCKING) ASYNC COMPILATION ###"
PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 ctest -T test --output-on-failure
echo "### END TESTING (BLOCKING) ASYNC COMPILATION ###"

# Test also our faster, alternative to HIP RTC codegen.
if [ "${CI_MACHINE}" == "tioga" ] && [ "${PROTEUS_CI_ROCM_VERSION}" == "6.2.1" ]; then
  echo "### TESTING SYNC COMPILATION WITH PROTEUS CODEGEN SERIAL ###"
  PROTEUS_CODEGEN=serial ctest -T test --output-on-failure
  echo "### END TESTING SYNC COMPILATION WITH PROTEUS HIP CODEGEN SERIAL ###"

  echo "### TESTING SYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL ###"
  PROTEUS_CODEGEN=parallel ctest -T test --output-on-failure
  echo "### END TESTING SYNC COMPILATION WITH PROTEUS HIP CODEGEN PARALLEL ###"

  echo "### TESTING SYNC COMPILATION WITH PROTEUS CODEGEN THINLTO ###"
  PROTEUS_CODEGEN=thinlto ctest -T test --output-on-failure
  echo "### END TESTING SYNC COMPILATION WITH PROTEUS HIP CODEGEN THINLTO ###"

  echo "### TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN SERIAL ###"
  PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=serial ctest -T test --output-on-failure
  echo "### END TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN SERIAL ###"

  echo "### TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL ###"
  PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=parallel ctest -T test --output-on-failure
  echo "### END TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL ###"

  echo "### TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN THINLTO ###"
  PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=thinlto ctest -T test --output-on-failure
  echo "### END TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN THINLTO ###"
fi

popd
