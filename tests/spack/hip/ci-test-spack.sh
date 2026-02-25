#!/bin/bash

set -e

ml load rocm/${PROTEUS_CI_ROCM_VERSION}

# Install spack.
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_USER_CACHE_PATH=/tmp/spack-${CI_JOB_ID}/user_cache
git clone --depth=2 https://github.com/spack/spack.git /tmp/spack-${CI_JOB_ID}
source /tmp/spack-${CI_JOB_ID}/share/spack/setup-env.sh

# Create environment.
spack env create -d /tmp/proteus-spack-env-${CI_JOB_ID}
spack env activate /tmp/proteus-spack-env-${CI_JOB_ID}

# Find externals.
spack external find
spack external find hip hsa-rocr-dev llvm-amdgpu

# Add repo and package.
spack repo add ${CI_PROJECT_DIR}/packaging/spack
spack add proteus@git.${CI_COMMIT_SHA} +rocm amdgpu_target=${PROTEUS_CI_AMDGPU_TARGET} ^hip@${PROTEUS_CI_ROCM_VERSION} ^hsa-rocr-dev@${PROTEUS_CI_ROCM_VERSION} ^llvm-amdgpu@${PROTEUS_CI_ROCM_VERSION}

# Concretize and install.
spack concretize -f
spack install -v

# Cleanup.
rm -rf ${SPACK_USER_CACHE_PATH}
rm -rf /tmp/proteus-spack-env-${CI_JOB_ID}
rm -rf /tmp/spack-${CI_JOB_ID}
