MPI_FLAGS =

CC?=hipcc
JIT_PATH ?= /p/vast1/ggeorgak/projects/compilers/jitproject/jit/build-tioga
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

LINKFLAGS =
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

# release flags
FLAGS += $(RFLAGS)

ifeq ($(JIT),yes)
  FLAGS += -fpass-plugin=$(JIT_PATH)/pass/libProteusJitPass.so -DENABLE_JIT
  LINKFLAGS += -Wl,-rpath,$(JIT_PATH)/lib -L${JIT_PATH}/lib/ -lproteusjit
endif

lulesh$(SUFFIX).x: allocator$(SUFFIX).o lulesh$(SUFFIX).o lulesh-comms$(SUFFIX).o lulesh-comms-gpu$(SUFFIX).o
	$(CC) $(LINKFLAGS) allocator$(SUFFIX).o lulesh$(SUFFIX).o lulesh-comms$(SUFFIX).o lulesh-comms-gpu$(SUFFIX).o -o $@

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
