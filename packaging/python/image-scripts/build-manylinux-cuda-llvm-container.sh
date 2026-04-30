#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

LLVM_VER="${LLVM_VER:-22.1.3}"
CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
CUDA_MAJOR_MINOR="${CUDA_MAJOR_MINOR:-12-4}"
BASE_IMAGE="${BASE_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/olympus-hpc/proteus-manylinux-cuda-llvm}"
IMAGE_TAG="${IMAGE_TAG:-${CUDA_VERSION}-${LLVM_VER}}"
IMAGE_REF="${IMAGE_REPO}:${IMAGE_TAG}"
PUSH="${PUSH:-1}"
CONTAINER_CLI="${CONTAINER_CLI:-docker}"

if ! command -v "${CONTAINER_CLI}" >/dev/null 2>&1; then
  echo "${CONTAINER_CLI} is required" >&2
  exit 1
fi

BUILD_CMD=(
  "${CONTAINER_CLI}" build
  --file "${REPO_ROOT}/packaging/python/image-scripts/manylinux-cuda-llvm.Dockerfile"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "LLVM_VER=${LLVM_VER}"
  --build-arg "CUDA_MAJOR_MINOR=${CUDA_MAJOR_MINOR}"
  --build-arg "CUDA_VERSION=${CUDA_VERSION}"
  --tag "${IMAGE_REF}"
)

BUILD_CMD+=("${REPO_ROOT}")

echo "Building ${IMAGE_REF}"
if [ "${PUSH}" = "1" ]; then
  echo "Push enabled. Ensure '${CONTAINER_CLI} login ghcr.io' has already succeeded."
else
  echo "Push disabled. The image will remain local in ${CONTAINER_CLI}."
fi

"${BUILD_CMD[@]}"

if [ "${PUSH}" = "1" ]; then
  "${CONTAINER_CLI}" push "${IMAGE_REF}"
fi

echo
echo "Built image: ${IMAGE_REF}"
