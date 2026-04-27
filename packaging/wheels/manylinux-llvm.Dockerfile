ARG BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64:2024.10.07-1
FROM ${BASE_IMAGE}

ARG LLVM_VER=22.1.3
LABEL org.opencontainers.image.source="https://github.com/Olympus-HPC/proteus"
LABEL org.opencontainers.image.description="manylinux_2_28 build image with LLVM/Clang/MLIR preinstalled for Proteus wheel builds"

ENV LLVM_VER=${LLVM_VER}
ENV PREFIX=/opt/llvm-${LLVM_VER}

COPY packaging/wheels/build-llvm-manylinux.sh /tmp/build-llvm-manylinux.sh

RUN bash /tmp/build-llvm-manylinux.sh \
 && rm -f /tmp/build-llvm-manylinux.sh \
 && rm -rf /tmp/llvm-project-${LLVM_VER}.src /tmp/llvm-project-${LLVM_VER}.src.tar.xz
