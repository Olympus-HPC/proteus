#!/bin/bash

set -e

ml load clang/${PROTEUS_CI_LLVM_VERSION}
ml load cuda/${PROTEUS_CI_CUDA_VERSION}

# Install spack.
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_USER_CACHE_PATH=/tmp/spack-${CI_JOB_ID}/user_cache
git clone --depth=2 https://github.com/spack/spack.git /tmp/spack-${CI_JOB_ID}
source /tmp/spack-${CI_JOB_ID}/share/spack/setup-env.sh

# Create environment.
spack env create -d /tmp/proteus-spack-env-${CI_JOB_ID}
spack env activate /tmp/proteus-spack-env-${CI_JOB_ID}

# Add external packages.
LLVM_PREFIX=$(llvm-config --prefix)
# We manually add llvm as an external package to avoid spack's detection logic
# which may return incompatible versions.
spack config add --file <(cat <<EOF
packages:
  llvm:
    buildable: false
    externals:
    - spec: "llvm@${PROTEUS_CI_LLVM_VERSION}+clang targets=all"
      prefix: ${LLVM_PREFIX}
      extra_attributes:
        compilers:
          c: ${LLVM_PREFIX}/bin/clang
          cxx: ${LLVM_PREFIX}/bin/clang++
EOF
)
spack external find
spack external find cuda

# Add repo and package.
spack repo add ${CI_PROJECT_DIR}/packaging/spack
spack add proteus@git.${CI_COMMIT_SHA} +cuda cuda_arch=${PROTEUS_CI_CUDA_ARCH} ^cuda@${PROTEUS_CI_CUDA_VERSION} ^llvm@${PROTEUS_CI_LLVM_VERSION}

# Concretize and install.
spack concretize -f
spack install -v

# Cleanup.
rm -rf ${SPACK_USER_CACHE_PATH}
rm -rf /tmp/proteus-spack-env-${CI_JOB_ID}
rm -rf /tmp/spack-${CI_JOB_ID}
