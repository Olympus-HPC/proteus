#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        ?= clang++
OPTIMIZE  = yes
DEBUG     = no
JIT_PATH  ?=/p/vast1/koparasy/JIT/jit/build-tioga/
JIT ?= no

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(JIT),yes)
  SUFFIX = "-jit"
else
  SUFFIX = "-aot"
endif

program = adam$(SUFFIX).x

source = main.cu
obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall

# Linker Flags
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lm

# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g
	LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3 -ffast-math
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
	$(CC) $(CFLAGS) -x hip --offload-arch=gfx90a -c $< -o $@

clean:
	rm -rf *.x *.o *.ll *.bc

run_no_jit: $(program_no_jit)
	$(LAUNCHER) ./$(program_no_jit) 10000 200 100

run_jit: $(program_with_jit)
	$(LAUNCHER) ./$(program_with_jit) 10000 200 100
