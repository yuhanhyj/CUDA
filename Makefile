# Compilers and commands
CC=             gcc
CXX=            gcc
NVCC=           nvcc
LINK=           nvcc
DEL_FILE=       rm -f

CFLAGS          = -O2 -w
NVCCFLAGS       = -O2 -arch=sm_60 -w

INCPATH         = .

OBJECTS		= radiator_cuda.o \
			radiator.o \
			main.o

TARGET		= radiator_cuda


all: radiator_cuda

radiator_cuda: main.o radiator_cuda.o radiator.o
	$(NVCC) $^ -o $(TARGET) -I$(INCPATH)

radiator_cuda.o: radiator_cuda.cu
	$(NVCC) radiator_cuda.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)