set -e

cmake -S ${CI_PROJECT_DIR} -B build-proteus \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_INSTALL_PREFIX=${PWD}/install-proteus \
    -DPROTEUS_ENABLE_HIP=on \
    -DENABLE_TESTS=off \
    -DBUILD_SHARED=on

pushd build-proteus
make -j install
popd

LLVM_LIBDIR=$(${LLVM_INSTALL_DIR}/bin/llvm-config --libdir)
LLVM_LIBS=$(${LLVM_INSTALL_DIR}/bin/llvm-config --libs)
LLVM_LIBS+=" "$(${LLVM_INSTALL_DIR}/bin/llvm-config --system-libs)

cp -r ${TEST_DIR}/* .
export PROTEUS_INSTALL_DIR=${PWD}/install-proteus
make -j
./main

