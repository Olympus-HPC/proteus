#!/bin/bash

set -e

ml load cuda/${PROTEUS_CI_CUDA_VERSION}
source /tmp/spack/share/spack/setup-env.sh

spack env create -d /tmp/proteus-spack-env-${CI_JOB_ID}
spack env activate /tmp/proteus-spack-env-${CI_JOB_ID}

spack external find
spack external find cuda

spack repo add ${CI_PROJECT_DIR}/packaging/spack
spack add proteus@main +cuda cuda_arch=${PROTEUS_CI_CUDA_ARCH} ^cuda@${PROTEUS_CI_CUDA_VERSION}

spack concretize -f
spack install -v

rm -rf /tmp/proteus-spack-env-${CI_JOB_ID}
