#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
CUDA_MAJOR_MINOR="${CUDA_MAJOR_MINOR:-12-4}"
BASE_IMAGE="${BASE_IMAGE:?BASE_IMAGE must point to the LLVM-enabled manylinux image}"
IMAGE_REPO="${IMAGE_REPO:-proteus-manylinux-cuda-local}"
IMAGE_TAG="${IMAGE_TAG:-${CUDA_VERSION}}"
IMAGE_REF="${IMAGE_REPO}:${IMAGE_TAG}"
PUSH="${PUSH:-0}"
CONTAINER_CLI="${CONTAINER_CLI:-docker}"

if ! command -v "${CONTAINER_CLI}" >/dev/null 2>&1; then
  echo "${CONTAINER_CLI} is required" >&2
  exit 1
fi

BUILD_CMD=(
  "${CONTAINER_CLI}" build
  --pull
  --file "${REPO_ROOT}/packaging/python/image-scripts/manylinux-cuda.Dockerfile"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "CUDA_MAJOR_MINOR=${CUDA_MAJOR_MINOR}"
  --build-arg "CUDA_VERSION=${CUDA_VERSION}"
  --tag "${IMAGE_REF}"
)

BUILD_CMD+=("${REPO_ROOT}")

echo "Building ${IMAGE_REF} from ${BASE_IMAGE}"
if [ "${PUSH}" = "1" ]; then
  echo "Push enabled. Ensure '${CONTAINER_CLI} login' has already succeeded."
else
  echo "Push disabled. The image will remain local in ${CONTAINER_CLI}."
fi

"${BUILD_CMD[@]}"

if [ "${PUSH}" = "1" ]; then
  "${CONTAINER_CLI}" push "${IMAGE_REF}"
fi

echo
echo "Built image: ${IMAGE_REF}"
