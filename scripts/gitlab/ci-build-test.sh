#!/bin/bash

set -e

echo "### Start ci-build-test $(date)"

mkdir -p /tmp/proteus-ci-${CI_JOB_ID}
cd /tmp/proteus-ci-${CI_JOB_ID}

if [ "${CI_MACHINE}" == "tioga" ]; then
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
CMAKE_OPTIONS+=" -DENABLE_DEVELOPER_COMPILER_FLAGS=on"
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
echo "### $(date) START TESTING SYNC COMPILATION KERNEL_CLONE cross-clone ###"
PROTEUS_KERNEL_CLONE=cross-clone ctest -j8 -T test --output-on-failure
echo "### $(date) END TESTING SYNC COMPILATION KERNEL_CLONE cross-clone ###"

echo "### $(date) START TESTING SYNC COMPILATION KERNEL_CLONE link-clone-light ###"
PROTEUS_KERNEL_CLONE=link-clone-light ctest -j8 -T test --output-on-failure
echo "### $(date) END TESTING SYNC COMPILATION KERNEL_CLONE link-clone-light ###"

echo "### $(date) START TESTING SYNC COMPILATION KERNEL_CLONE link-clone-prune ###"
PROTEUS_KERNEL_CLONE=link-clone-prune ctest -j8 -T test --output-on-failure
echo "### $(date) END TESTING SYNC COMPILATION KERNEL_CLONE link-clone-prune ###"

# Test asynchronous compilation.
echo "### $(date) START TESTING (BLOCKING) ASYNC COMPILATION ###"
PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 ctest -j8 -T test --output-on-failure
echo "### $(date) END TESTING (BLOCKING) ASYNC COMPILATION ###"

# Test also our faster, alternative to HIP RTC codegen.
if [ "${CI_MACHINE}" == "tioga" ]; then
  echo "### $(date) START TESTING SYNC COMPILATION WITH PROTEUS CODEGEN SERIAL ###"
  PROTEUS_CODEGEN=serial ctest -j8 -T test --output-on-failure
  echo "### $(date) END TESTING SYNC COMPILATION WITH PROTEUS HIP CODEGEN SERIAL ###"

  echo "### $(date) START TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN SERIAL ###"
  PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=serial ctest -j8 -T test --output-on-failure
  echo "### $(date) END TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN SERIAL ###"

  if [ "${PROTEUS_CI_ROCM_VERSION}" == "6.2.1" ] || [ "${PROTEUS_CI_ROCM_VERSION}" == "6.3.1" ];  then
    echo "### $(date) START TESTING SYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL ###"
    PROTEUS_CODEGEN=parallel ctest -j8 -T test --output-on-failure
    echo "### $(date) END TESTING SYNC COMPILATION WITH PROTEUS HIP CODEGEN PARALLEL ###"

    echo "### $(date) START TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL ###"
    PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=parallel ctest -j8 -T test --output-on-failure
    echo "### $(date) END TESTING (BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL ###"
  fi
fi

popd
