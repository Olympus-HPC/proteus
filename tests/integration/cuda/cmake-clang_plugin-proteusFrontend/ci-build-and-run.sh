set -e

cmake -S ${CI_PROJECT_DIR} -B build-proteus \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_INSTALL_PREFIX=${PWD}/install-proteus \
    -DPROTEUS_ENABLE_CUDA=on \
    -DENABLE_TESTS=off \
    -DBUILD_SHARED=off

pushd build-proteus
make -j install
popd

cmake -S ${TEST_DIR} -B build \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_CUDA_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -Dproteus_DIR="${PWD}/install-proteus" \
    -DCMAKE_INSTALL_PREFIX="${PWD}/install"
pushd build
make -j install VERBOSE=1
popd

# Run from install directory.
pushd install/bin;
if ./main | grep -q " ==> Clang Plugin running"; then
    echo "Test passed: output contains '==> Clang Plugin running'"
else
    echo "Test failed: output does not contain '==> Clang Plugin running'"
    exit 1
fi
popd
