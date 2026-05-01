ARG BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64
FROM ${BASE_IMAGE}

ARG LLVM_VER=22.1.3
ARG CUDA_MAJOR_MINOR=12-4
ARG CUDA_VERSION=12.4.1
LABEL org.opencontainers.image.source="https://github.com/Olympus-HPC/proteus"
LABEL org.opencontainers.image.description="manylinux_2_28 build image with LLVM/Clang and CUDA preinstalled for Proteus CUDA wheel builds"

ENV LLVM_VER=${LLVM_VER}
ENV PREFIX=/opt/llvm-${LLVM_VER}
ENV CUDA_HOME=/usr/local/cuda

COPY packaging/python/image-scripts/build-llvm-manylinux.sh /tmp/build-llvm-manylinux.sh

RUN bash /tmp/build-llvm-manylinux.sh \
 && rm -f /tmp/build-llvm-manylinux.sh \
 && rm -rf /tmp/llvm-project-${LLVM_VER}.src /tmp/llvm-project-${LLVM_VER}.src.tar.xz

RUN dnf install -y 'dnf-command(config-manager)' \
 && dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo" \
 && dnf clean all \
 && dnf install -y "cuda-toolkit-${CUDA_MAJOR_MINOR}" \
 && test -f /usr/local/cuda/lib64/libnvptxcompiler_static.a \
 && dnf clean all
