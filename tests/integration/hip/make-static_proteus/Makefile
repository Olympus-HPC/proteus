all:
	hipcc -fpass-plugin=${PROTEUS_INSTALL_DIR}/lib64/libProteusPass.so main.cpp -o main \
		-I${PROTEUS_INSTALL_DIR}/include \
		-L${PROTEUS_INSTALL_DIR}/lib64  \
		-Wl,-rpath,${PROTEUS_INSTALL_DIR}/lib64 \
		-lproteus \
		${LLVM_LIBFLAGS}
