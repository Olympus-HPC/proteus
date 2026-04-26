#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-/opt/miniforge}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${MINIFORGE_PREFIX}/envs/proteus-llvm22}"
MINIFORGE_VER="${MINIFORGE_VER:-25.3.1-0}"
MINIFORGE_INSTALLER="${MINIFORGE_INSTALLER:-/tmp/Miniforge3-${MINIFORGE_VER}-Linux-x86_64.sh}"
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VER}/Miniforge3-${MINIFORGE_VER}-Linux-x86_64.sh}"

if [ -x "${CONDA_ENV_PREFIX}/bin/clang++" ]; then
  echo "Reusing existing Miniforge LLVM toolchain at ${CONDA_ENV_PREFIX}"
  exit 0
fi

mkdir -p /tmp

if [ ! -x "${MINIFORGE_PREFIX}/bin/conda" ]; then
  curl -LsSf "${MINIFORGE_URL}" -o "${MINIFORGE_INSTALLER}"
  bash "${MINIFORGE_INSTALLER}" -b -p "${MINIFORGE_PREFIX}"
fi

"${MINIFORGE_PREFIX}/bin/conda" create -y -p "${CONDA_ENV_PREFIX}" -c conda-forge \
  sysroot_linux-64=2.28 \
  llvmdev=22.1.* \
  mlir=22.1.* \
  clang=22.1.* \
  clangdev=22.1.* \
  cmake \
  ninja \
  zlib \
  zstd \
  libxml2

echo "Installed Miniforge LLVM toolchain to: ${CONDA_ENV_PREFIX}"
