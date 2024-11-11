#===============================================================================
# User Options
#===============================================================================

CC ?= clang
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
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

program = rsbench$(SUFFIX).x

source = \
main.cpp \
simulation.cpp\
io.cpp \
init.cpp \
material.cpp \
utils.cpp

obj = $(source:.cpp=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS :=

CFLAGS += -std=c++14

# Linker Flags
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
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

$(program): $(obj) rsbench.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cpp rsbench.h Makefile
	$(CC) $(CFLAGS) -x hip --offload-arch=gfx90a -c $< -o $@

clean:
	rm -rf *.x *.o *.ll *.bc
edit:
	vim -p $(source) rsbench.h

run:
	./rsbench
