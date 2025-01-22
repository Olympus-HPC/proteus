#!/bin/sh

if [ $# -eq 0 ]; then
    ROCM_VERSION="5.7.1"
else
    ROCM_VERSION=$1
fi

ml load rocm/${ROCM_VERSION}

LLVM_INSTALL_DIR=${ROCM_PATH}/llvm

mkdir build-rocm-${ROCM_VERSION}
pushd build-rocm-${ROCM_VERSION}

cmake .. \
-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
-DPROTEUS_ENABLE_HIP=On \
-DCMAKE_INSTALL_PREFIX=../install-rocm-${ROCM_VERSION} \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on

popd
