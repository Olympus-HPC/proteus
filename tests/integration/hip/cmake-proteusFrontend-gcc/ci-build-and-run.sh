set -e

cmake -S ${CI_PROJECT_DIR} -B build-proteus \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DCMAKE_CXX_COMPILER="gcc" \
    -DCMAKE_CXX_COMPILER="g++" \
    -DCMAKE_INSTALL_PREFIX=${PWD}/install-proteus \
    -DPROTEUS_ENABLE_HIP=on \
    -DENABLE_TESTS=off \
    -DBUILD_SHARED=off

pushd build-proteus
make -j install
popd

cmake -S ${TEST_DIR} -B build \
    -DCMAKE_CXX_COMPILER="g++" \
    -Dproteus_DIR="${PWD}/install-proteus" \
    -DCMAKE_INSTALL_PREFIX="${PWD}/install"
pushd build
make -j install
popd

# Run from install directory.
install/bin/main
