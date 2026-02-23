#!/bin/bash

set -e

echo "Run Integration Test ${PROTEUS_CI_INTEGRATION_TEST}"

# Use pre-installed LLVM 20 on matrix.
export LLVM_INSTALL_DIR=/usr/lib64/llvm20
echo "LLVM_INSTALL_DIR ${LLVM_INSTALL_DIR}"

# Load cuda module.
ml load cuda/12

export TEST_DIR=${CI_PROJECT_DIR}/tests/integration/cuda/${PROTEUS_CI_INTEGRATION_TEST}

mkdir -p /tmp/proteus-ci-$(basename ${TEST_DIR})-${CI_JOB_ID}
pushd /tmp/proteus-ci-$(basename ${TEST_DIR})-${CI_JOB_ID}

rm -rf build-proteus install-proteus build install
bash ${TEST_DIR}/ci-build-and-run.sh

popd

echo "=> Passed ${PROTEUS_CI_INTEGRATION_TEST}"
