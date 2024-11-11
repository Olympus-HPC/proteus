#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
# Compiler can be set below, or via environment variable
CC        ?= amdclang++
OPTIMIZE  = yes
DEBUG     = no
JIT_PATH  ?= /p/vast1/ggeorgak/projects/compilers/jitproject/jit/build-tioga
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
CFLAGS := --offload-arch=gfx90a:sramecc+:xnack- $(EXTRA_CFLAGS) -std=c++14 -Wall

# Linker Flags
LDFLAGS =
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lm

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
  CFLAGS += -fpass-plugin=$(JIT_PATH)/pass/libProteusJitPass.so -DENABLE_JIT
  LDFLAGS += -Wl,-rpath,$(JIT_PATH)/lib -L${JIT_PATH}/lib/ -lproteusjit
endif

#===============================================================================
# Targets to Build
#===============================================================================
$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu Makefile
	$(CC) $(CFLAGS) -x hip  -c $< -o $@

clean:
	rm -rf *.x *.o *.ll *.bc

run: $(program)
	$(LAUNCHER) ./$(program) ../sw4ck-cuda/sw4ck.in 100
