MPI_FLAGS =

CC?=nvcc
ARCH=sm_70
JIT_PATH ?= /p/vast1/ggeorgak/projects/compilers/jitproject/jit/build-lassen
JIT ?= no
FLAGS	=
DFLAGS	= $(MPI_FLAGS) -lineinfo
RFLAGS 	= $(MPI_FLAGS) -O3 -DNDEBUG

ifeq ($(JIT),yes)
  SUFFIX = "-jit"
else
  SUFFIX = "-aot"
endif


#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

LINKFLAGS = -Wl,-rpath,${CUDA_HOME}/nvidia/lib64 -L${CUDA_HOME}/nvidia/lib64 \
	  -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

# release flags
FLAGS += --cuda-gpu-arch=$(ARCH) --cuda-path=${CUDA_HOME}/nvidia $(RFLAGS)

ifeq ($(JIT),yes)
  LIBLLVM_PATH=$(dir $(realpath $(CC)))../lib
  FLAGS += -fpass-plugin=$(JIT_PATH)/pass/libProteusJitPass.so -DENABLE_JIT
  LINKFLAGS += -Wl,-rpath,$(LIBLLVM_PATH) -L$(LIBLLVM_PATH) -lLLVM \
	       -Wl,-rpath,$(JIT_PATH)/lib -L${JIT_PATH}/lib/ -lproteusjit -lnvptxcompiler_static
endif

lulesh$(SUFFIX).x: allocator$(SUFFIX).o lulesh$(SUFFIX).o lulesh-comms$(SUFFIX).o lulesh-comms-gpu$(SUFFIX).o
	$(CC) allocator$(SUFFIX).o lulesh$(SUFFIX).o lulesh-comms$(SUFFIX).o lulesh-comms-gpu$(SUFFIX).o -o $@ $(LINKFLAGS)

allocator$(SUFFIX).o: allocator.cu vector.h
	$(CC) $(FLAGS) allocator.cu -I ./ -c -o $@

lulesh$(SUFFIX).o: lulesh.cu util.h vector.h allocator.h
	$(CC) $(FLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o $@

lulesh-comms$(SUFFIX).o: lulesh-comms.cu
	$(CC) $(FLAGS) lulesh-comms.cu -I ./ -c -o $@

lulesh-comms-gpu$(SUFFIX).o: lulesh-comms-gpu.cu
	$(CC) $(FLAGS) lulesh-comms-gpu.cu -I ./ -c -o $@

clean:
	rm -rf *.x *.o xyz.asc regNumList.txt
