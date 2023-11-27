#!/bin/bash

trap exit INT

ml load cmake/3.23.1
ml load ninja
ml load gcc/11.2.1
ml load cuda/11.8

mkdir -p build-lassen-llvm-17.0.5
pushd build-lassen-llvm-17.0.5

if ! [ -f "llvmorg-17.0.5.tar.gz" ]; then
    echo "===> Get llvm tarball..."
    wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-17.0.5.tar.gz
fi

echo "===> Decompress llvm tarball..."
pigz -dc llvmorg-17.0.5.tar.gz | tar --skip-old-files -xf - || (echo "===> Error decompressing"; exit 1)

BUILD_TYPE=RelWithDebInfo
CMAKEDIR=llvm-project-llvmorg-17.0.5/llvm
ASSERTIONS=on

cmake -G Ninja \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_PROJECTS='clang;lld' \
    -DCMAKE_INSTALL_PREFIX=${PWD}/install \
    -DLLVM_TARGETS_TO_BUILD="PowerPC;NVPTX" \
    -DLLVM_INSTALL_UTILS=on \
    -DLLVM_LINK_LLVM_DYLIB=on \
    -DLLVM_ENABLE_ASSERTIONS=${ASSERTIONS} \
    -DLLVM_USE_LINKER=gold \
    ${CMAKEDIR}

echo "===> Compile llvm..."
ninja 

echo "===> Install llvm..."
ninja install

popd
