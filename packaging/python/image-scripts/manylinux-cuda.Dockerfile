ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_MAJOR_MINOR=12-4
ARG CUDA_VERSION=12.4.1
LABEL org.opencontainers.image.source="https://github.com/Olympus-HPC/proteus"
LABEL org.opencontainers.image.description="manylinux_2_28 build image with CUDA preinstalled for Proteus CUDA wheel builds"

ENV CUDA_HOME=/usr/local/cuda

RUN dnf install -y 'dnf-command(config-manager)' \
 && dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo" \
 && dnf clean all \
 && dnf install -y "cuda-toolkit-${CUDA_MAJOR_MINOR}" \
 && test -f /usr/local/cuda/lib64/libnvptxcompiler_static.a \
 && dnf clean all \
 && rm -rf /var/cache/dnf
