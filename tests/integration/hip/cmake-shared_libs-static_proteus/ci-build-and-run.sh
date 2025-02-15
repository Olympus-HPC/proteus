set -e

cat <<EOF
*********************************************************
 This test expectedly fails because of static LLVM init,
 see https://github.com/Olympus-HPC/proteus/issues/134
*********************************************************
EOF
exit 0

cmake -S ${CI_PROJECT_DIR} -B build-proteus \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
    -DCMAKE_INSTALL_PREFIX=${PWD}/install-proteus \
    -DPROTEUS_ENABLE_HIP=on \
    -DENABLE_TESTS=off \
    -DBUILD_SHARED=off \
    -DCMAKE_POSITION_INDEPENDENT_CODE=on

pushd build-proteus
make -j install
popd

cmake -S ${TEST_DIR} -B build \
    -Dproteus_DIR="${PWD}/install-proteus" \
    -DCMAKE_INSTALL_PREFIX="${PWD}/install"
pushd build
make -j install
# Run from build directory.
./main
popd

install/bin/main
