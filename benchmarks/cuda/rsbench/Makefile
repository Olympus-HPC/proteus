#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        ?= nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
LAUNCHER  =
JIT_PATH    ?=/p/vast1/koparasy/JIT/jit/build-lassen/
JIT ?= no

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(JIT),yes)
  SUFFIX = "-jit"
else
  SUFFIX = "-aot"
endif

program = rsbench$(SUFFIX).x

source = \
main.cu \
simulation.cu\
io.cu \
init.cu \
material.cu \
utils.cu

obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 --cuda-gpu-arch=$(ARCH) --cuda-path=${CUDA_HOME}/nvidia

# Linker Flags
LDFLAGS = -Wl,-rpath,${CUDA_HOME}/nvidia/lib64 -L${CUDA_HOME}/nvidia/lib64 \
	  -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(JIT),yes)
  LIBLLVM_PATH=$(dir $(realpath $(CC)))../lib
  CFLAGS += -fpass-plugin=$(JIT_PATH)/pass/libProteusJitPass.so -DENABLE_JIT
  LDFLAGS += -Wl,-rpath,$(LIBLLVM_PATH) -L$(LIBLLVM_PATH) -lLLVM \
	     -Wl,-rpath,$(JIT_PATH)/lib -L$(JIT_PATH)/lib/ -lproteusjit -lnvptxcompiler_static
endif

LDFLAGS +=-lnvptxcompiler_static

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) rsbench.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu rsbench.h Makefile
	$(CC) $(CFLAGS) -c  $< -o $@

clean:
	rm -rf *.x *.o

run: $(program)
	$(LAUNCHER) ./$(program) -s large -m event
