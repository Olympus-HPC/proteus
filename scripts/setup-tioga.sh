#-DLLVM_INSTALL_DIR=/g/g11/bowen36/llvm-install-rti-release \
#-DLLVM_INSTALL_DIR=/opt/rocm-6.1.1/llvm/ \
module load rocm/6.1.1

mkdir build-tioga
pushd build-tioga

cmake .. \
-DLLVM_INSTALL_DIR=/opt/rocm-6.1.1/llvm \
-DLLVM_VERSION=17 \
-DCMAKE_C_COMPILER=amdclang \
-DCMAKE_CXX_COMPILER=amdclang++ \
-DENABLE_HIP=on \
-DBUILD_SHARED_LIBJIT=On \
-DLLVM_ENABLE_RTTI=OFF \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on \
-DCMAKE_BUILD_TYPE=DEBUG \
-DBUILD_SHARED_LIBS=On \
-DCMAKE_INSTALL_PREFIX=~/tmp \
-DCMAKE_HIP_ARCHITECTURES=gfx90a \

popd
