# NOTE: load any needed modules for ROCm

LLVM_INSTALL_DIR=${ROCM_PATH}/llvm

rm -rf build-amd
mkdir build-amd
pushd build-amd

cmake .. \
-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
-DLLVM_VERSION=17 \
-DBUILD_SHARED_LIBJIT=on \
-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
-DENABLE_HIP=on \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on

popd
