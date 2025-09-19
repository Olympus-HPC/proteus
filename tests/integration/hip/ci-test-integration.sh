#!/bin/bash

set -e

echo "Run Integration Test ${PROTEUS_CI_INTEGRATION_TEST}"

ml load rocm/${PROTEUS_CI_ROCM_VERSION}
if [ -z "${PROTEUS_CI_ROCM_VERSION}" ]; then
    echo "Expected non-empty PROTEUS_CI_ROCM_VERSION env var"
    exit 1
fi
echo "PROTEUS_CI_ROCM_VERSION ${PROTEUS_CI_ROCM_VERSION}"

export LLVM_INSTALL_DIR=$(realpath ${ROCM_PATH}/llvm)
if [ -z "${LLVM_INSTALL_DIR}" ]; then
    echo "Expected non-empty LLVM_INSTALL_DIR env var"
    exit 1
fi
echo "LLVM_INSTALL_DIR ${LLVM_INSTALL_DIR}"

export TEST_DIR=${CI_PROJECT_DIR}/tests/integration/hip/${PROTEUS_CI_INTEGRATION_TEST}

mkdir -p /tmp/proteus-ci-$(basename ${TEST_DIR})-${CI_JOB_ID}
pushd /tmp/proteus-ci-$(basename ${TEST_DIR})-${CI_JOB_ID}

rm -rf build-proteus install-proteus build
bash ${TEST_DIR}/ci-build-and-run.sh

popd

echo "=> Passed ${PROTEUS_CI_INTEGRATION_TEST}"
