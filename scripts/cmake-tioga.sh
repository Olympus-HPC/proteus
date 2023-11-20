ml load rocm/5.7.1

mkdir build-tioga
pushd build-tioga

cmake .. \
-DLLVM_INSTALL_DIR=/opt/rocm-5.7.1/llvm \
-DLLVM_VERSION=17 \
-DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ -DENABLE_HIP=on \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on

popd
