#!/bin/bash

set -e

ml load rocm/${PROTEUS_CI_ROCM_VERSION}
source /tmp/spack/share/spack/setup-env.sh

spack env create -d /tmp/proteus-spack-env-${CI_JOB_ID}
spack env activate /tmp/proteus-spack-env-${CI_JOB_ID}

spack external find
spack external find hip hsa-rocr-dev llvm-amdgpu

spack repo add ${CI_PROJECT_DIR}/packaging/spack
spack add proteus@git.${CI_COMMIT_SHA} +rocm amdgpu_target=${PROTEUS_CI_AMDGPU_TARGET} ^hip@${PROTEUS_CI_ROCM_VERSION} ^hsa-rocr-dev@${PROTEUS_CI_ROCM_VERSION} ^llvm-amdgpu@${PROTEUS_CI_ROCM_VERSION}

spack concretize -f
spack install -v

rm -rf /tmp/proteus-spack-env-${CI_JOB_ID}
