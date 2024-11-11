#!/bin/bash

function error() {
  echo "Error occured: $1"
}

function setup() {
  # Check if script is sourced.
  if [ -n "$BASH_SOURCE" ] && [ "$BASH_SOURCE" = "$0" ]; then
    echo "This script MUST BE sourced from the root directory, do: source $0"
    return 1
  fi

  if [ -n "$CUDA_HOME" ] || [ -n "$CUDA_PATH" ]; then
      echo "=> Detected CUDA"
      EXTRA_PKGS="clang=17.0.5 clangxx=17.0.5 llvmdev=17.0.5 lit=17.0.5"
      CMAKE_SCRIPT=cmake-nvidia.sh
      VENDOR=nvidia
  elif [ -n "$ROCM_PATH" ]; then
      echo "=> Detected ROCm"
      EXTRA_PKGS="llvmdev=17.0.5 lit=17.0.5"
      CMAKE_SCRIPT=cmake-amd.sh
      VENDOR=amd
  else
    echo "=> Failed to detect CUDA or ROCm installation. Check your environment and try again."
    return 1
  fi

  # Install miniconda to setup the environment.
  source buildscripts/install-miniconda3.sh

  if conda activate proteus &> /dev/null; then
    echo "Environment proteus already exists and activated"
  else
    echo "Creating environment proteus..."
    conda create -y -n proteus -c conda-forge \
      python=3.10 cmake=3.24.3 pandas cxxfilt matplotlib ${EXTRA_PKGS} || { error $?; return 1; }

    conda activate proteus || { error $?; return 1; }
  fi

  # Fix to expose the FileCheck executable, needed for building Proteus.
  if [ ! -f ${CONDA_PREFIX}/bin/FileCheck ]; then
    ln -s ${CONDA_PREFIX}/libexec/llvm/FileCheck ${CONDA_PREFIX}/bin
  fi

  # Build Proteus.
  bash buildscripts/${CMAKE_SCRIPT} || { error $?; return 1; }
  pushd build-${VENDOR}
  make -j || { error $?; return 1; }
  popd
}

setup
