#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        ?= nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
LAUNCHER  =
JIT_PATH    ?=/p/vast1/koparasy/JIT/jit/build/
JIT ?= no

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(JIT),yes)
  SUFFIX = "-jit"
else
  SUFFIX = "-aot"
endif

program = sw4ck$(SUFFIX).x

source = main.cu curvilinear4sg.cu

obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 --cuda-gpu-arch=$(ARCH) --cuda-path=${CUDA_HOME}/nvidia

# Linker Flags
LDFLAGS = -Wl,-rpath,${CUDA_HOME}/nvidia/lib64 -L${CUDA_HOME}/nvidia/lib64 \
	  -lnvrtc -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

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
  LDFLAGS +=  -lnvptxcompiler_static
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main$(SUFFIX).o: main.cu utils.cpp utils.h
	$(CC) $(CFLAGS) -c $< -o $@

curvilinear4sg$(SUFFIX).o : curvilinear4sg.cu \
       	           kernel1.cpp \
	           kernel2.cpp \
       	           kernel3.cpp \
	           kernel4.cpp \
	           kernel5.cpp \
	           utils.h \
	           curvilinear4sg.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.x *.o

run: $(program)
	$(LAUNCHER) ./$(program) sw4ck.in 100
