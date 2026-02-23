OFFLOAD_ARCH=sm_90

all:
	clang++ -x cu -std=c++17 -fpass-plugin=${PROTEUS_INSTALL_DIR}/lib64/libProteusPass.so main.cpp -o main \
		-I${PROTEUS_INSTALL_DIR}/include \
		--offload-arch=${OFFLOAD_ARCH} \
		-I${CUDA_HOME}/include \
		-L${PROTEUS_INSTALL_DIR}/lib64  \
		-Wl,-rpath,${PROTEUS_INSTALL_DIR}/lib64 \
		-l:libproteus.a \
		-l:libproteus_cudart_builtins.a \
		${LLVM_LIBFLAGS} \
		-L${CUDA_HOME}/lib64 \
		-lcuda \
		-lcudart \
		-pthread \
		-lnvptxcompiler_static
