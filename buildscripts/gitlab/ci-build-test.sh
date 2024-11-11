#!/bin/bash

set -e

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

  # Fix to expose the FileCheck executable, needed for building proteus tests.
  ln -s ${CONDA_PREFIX}/libexec/llvm/FileCheck ${CONDA_PREFIX}/bin

  LLVM_INSTALL_DIR=$(llvm-config --prefix)
  CMAKE_OPTIONS_MACHINE=" -DENABLE_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=$LLVM_INSTALL_DIR/bin/clang++"
elif [ "${CI_MACHINE}" == "tioga" ]; then
  ml load rocm/5.7.1

  LLVM_INSTALL_DIR=/opt/rocm-5.7.1/llvm
  CMAKE_OPTIONS_MACHINE=" -DENABLE_HIP=on"
else
  echo "Unsupported machine ${CI_MACHINE}"
  exit 1
fi

CMAKE_OPTIONS="-DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR -DLLVM_VERSION=17"
CMAKE_OPTIONS+=" -DCMAKE_C_COMPILER=$LLVM_INSTALL_DIR/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_INSTALL_DIR/bin/clang++"
CMAKE_OPTIONS+=${CMAKE_OPTIONS_MACHINE}
CMAKE_OPTIONS+=" -DENABLE_DEBUG=${PROTEUS_CI_ENABLE_DEBUG} -DENABLE_TIME_TRACING=${PROTEUS_CI_ENABLE_TIME_TRACING}"
if [ -n "${PROTEUS_CI_BUILD_SHARED}" ]; then
  CMAKE_OPTIONS+=" -DBUILD_SHARED=${PROTEUS_CI_BUILD_SHARED}"
fi

mkdir build
pushd build

cmake .. ${CMAKE_OPTIONS}
make -j
make test

popd