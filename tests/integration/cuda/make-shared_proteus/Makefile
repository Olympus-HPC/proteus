OFFLOAD_ARCH=sm_90

all:
	clang++ -x cu -std=c++17 -fpass-plugin=${PROTEUS_INSTALL_DIR}/lib64/libProteusPass.so main.cpp -o main \
		--offload-arch=${OFFLOAD_ARCH} \
		-I${PROTEUS_INSTALL_DIR}/include \
		-I${CUDA_HOME}/include \
		-L${PROTEUS_INSTALL_DIR}/lib64  \
		-Wl,-rpath,${PROTEUS_INSTALL_DIR}/lib64 \
		-l:libproteus.so \
		-l:libproteus_cudart_builtins.a \
		-L${CUDA_HOME}/lib64 \
		-lcuda \
		-lcudart
