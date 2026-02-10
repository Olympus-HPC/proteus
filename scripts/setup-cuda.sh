#!/bin/sh

HOSTN=${HOSTNAME//[0-9]/}

ml load cmake/3.23.1

LLVM_INSTALL_DIR=$1
CUDA_VERSION=$2
if [ $# -lt 2 ] || [ -z "${LLVM_INSTALL_DIR}" ] || [ -z "${CUDA_VERSION}" ];
then
    echo "Usage: source setup-cuda.sh <LLVM installation dir> <CUDA version>"
    return 0
fi

ml load cuda/${CUDA_VERSION}

export PATH="$LLVM_INSTALL_DIR/bin":$PATH

mkdir build-${HOSTN}-cuda-${CUDA_VERSION}
pushd build-${HOSTN}-cuda-${CUDA_VERSION}

cmake .. \
-DLLVM_INSTALL_DIR="$LLVM_INSTALL_DIR" \
-DPROTEUS_ENABLE_CUDA=on \
-DCMAKE_CUDA_ARCHITECTURES=90 \
-DCMAKE_C_COMPILER="$LLVM_INSTALL_DIR/bin/clang" \
-DCMAKE_CXX_COMPILER="$LLVM_INSTALL_DIR/bin/clang++" \
-DCMAKE_CUDA_COMPILER="$LLVM_INSTALL_DIR/bin/clang++" \
-DCMAKE_INSTALL_PREFIX=../install-${HOSTN}-cuda-${CUDA_VERSION} \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on \
"${@:3}"

popd
