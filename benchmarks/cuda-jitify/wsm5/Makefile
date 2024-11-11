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

program = main$(SUFFIX).x

source = main.cu

obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DMKX=4 -std=c++14 -Xcompiler -Wall --offload-arch=$(ARCH) --cuda-path=${CUDA_HOME}/nvidia

# Linker Flags
LDFLAGS = -Wl,-rpath,${CUDA_HOME}/nvidia/lib64 -L${CUDA_HOME}/nvidia/lib64 \
	  -lnvrtc -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl


# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(JIT),yes)
  LIBLLVM_PATH=$(dir $(realpath $(CC)))../lib
  CFLAGS += -DENABLE_JIT
  LDFLAGS += -lnvptxcompiler_static
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu kernel.h spt.h constants.h utils.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 10
