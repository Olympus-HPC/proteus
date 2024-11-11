# NOTE: load any needed modules for CUDA

LLVM_INSTALL_DIR=$(llvm-config --prefix)

rm -rf build-nvidia
mkdir build-nvidia
pushd build-nvidia

cmake .. \
-DLLVM_INSTALL_DIR="$LLVM_INSTALL_DIR" \
-DLLVM_VERSION=17 \
-DENABLE_CUDA=on \
-DCMAKE_CUDA_ARCHITECTURES=70 \
-DCMAKE_C_COMPILER="$LLVM_INSTALL_DIR/bin/clang" \
-DCMAKE_CXX_COMPILER="$LLVM_INSTALL_DIR/bin/clang++" \
-DCMAKE_CUDA_COMPILER="$LLVM_INSTALL_DIR/bin/clang++" \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on

popd

