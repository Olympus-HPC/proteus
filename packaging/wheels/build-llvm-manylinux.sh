#!/usr/bin/env bash
set -euo pipefail

LLVM_VER="${LLVM_VER:-22.1.3}"
PREFIX="${PREFIX:-/opt/llvm-${LLVM_VER}}"
WORKDIR="${WORKDIR:-/tmp}"
TARBALL="llvm-project-${LLVM_VER}.src.tar.xz"
SRCDIR="llvm-project-${LLVM_VER}.src"
URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/${TARBALL}"

if [ -x "${PREFIX}/bin/clang++" ]; then
  echo "Reusing existing LLVM toolchain at ${PREFIX}"
  exit 0
fi

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

yum install -y \
  cmake \
  curl \
  gcc \
  gcc-c++ \
  git \
  libffi-devel \
  make \
  python3 \
  python3-pip \
  tar \
  xz \
  zlib-devel

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade ninja

[ -f "${TARBALL}" ] || curl -LO "${URL}"
[ -d "${SRCDIR}" ] || tar -xf "${TARBALL}"

cd "${SRCDIR}"
rm -rf build

cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_ENABLE_RUNTIMES="" \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_BUILD_UTILS=OFF \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DCLANG_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF

cmake --build build --parallel "$(getconf _NPROCESSORS_ONLN)" --target install

test -f "${PREFIX}/lib/cmake/llvm/LLVMConfig.cmake"
test -f "${PREFIX}/lib/cmake/clang/ClangConfig.cmake"
test -f "${PREFIX}/lib/libLLVMCore.a"
test -f "${PREFIX}/lib/libclangFrontend.a"

echo
echo "Installed LLVM ${LLVM_VER} to: ${PREFIX}"
