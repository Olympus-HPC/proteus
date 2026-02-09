set -e

cmake -S ${CI_PROJECT_DIR} -B build-proteus \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_INSTALL_PREFIX=${PWD}/install-proteus \
    -DPROTEUS_ENABLE_HIP=on \
    -DPROTEUS_INSTALL_IMPL_HEADERS=on \
    -DENABLE_TESTS=off \
    -DBUILD_SHARED=off

pushd build-proteus
make -j install
popd

cmake -S ${TEST_DIR} -B build \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -Dproteus_DIR="${PWD}/install-proteus" \
    -DCMAKE_INSTALL_PREFIX="${PWD}/install"
pushd build
make -j install
popd

# Run from install directory.
install/bin/main build/test.bc foo
