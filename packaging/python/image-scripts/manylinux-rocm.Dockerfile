ARG BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64
FROM ${BASE_IMAGE}

ARG ROCM_VERSION=7.2.1
ARG ROCM_EL_VERSION=8
LABEL org.opencontainers.image.source="https://github.com/Olympus-HPC/proteus"
LABEL org.opencontainers.image.description="manylinux_2_28 build image with ROCm preinstalled for Proteus ROCm wheel builds"

RUN cat > /etc/yum.repos.d/rocm.repo <<EOF
[rocm]
name=ROCm ${ROCM_VERSION} repository
baseurl=https://repo.radeon.com/rocm/el${ROCM_EL_VERSION}/${ROCM_VERSION}/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

RUN dnf clean all \
 && dnf install -y --setopt=install_weak_deps=False libzstd-devel rocm-hip-runtime-devel rocm-llvm-devel \
 && if [ ! -e /opt/rocm/llvm ] && [ -d "/opt/rocm-${ROCM_VERSION}/lib/llvm" ]; then ln -s "/opt/rocm-${ROCM_VERSION}/lib/llvm" /opt/rocm/llvm; fi \
 && test -x /opt/rocm/bin/hipcc \
 && test -x "/opt/rocm-${ROCM_VERSION}/lib/llvm/bin/amdclang++" \
 && test -f "/opt/rocm-${ROCM_VERSION}/lib/llvm/lib/cmake/llvm/LLVMConfig.cmake" \
 && dnf clean all \
 && rm -rf /var/cache/dnf
