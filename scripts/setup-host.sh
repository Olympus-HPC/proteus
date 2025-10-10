#!/bin/sh

CLANG_VERSION=$1
if [ $# -lt 1 ]; then
    echo "Usage: source setup-rocm.sh <Clang version>"
    return 0
fi

#ml load clang/${CLANG_VERSION}

LLVM_INSTALL_DIR=$(llvm-config --prefix)

mkdir build-host-clang_${CLANG_VERSION}
pushd build-host-clang_${CLANG_VERSION}
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:${CMAKE_PREFIX_PATH}"

cmake .. \
-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
-DCMAKE_INSTALL_PREFIX=../install-rocm-${CLANG_VERSION} \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on \
-DPROTEUS_LINK_SHARED_LLVM=on \
"${@:2}"

popd
