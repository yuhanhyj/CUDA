# Compilers and commands
CC=             gcc
CXX=            gcc
NVCC=           nvcc
LINK=           nvcc
DEL_FILE=       rm -f

CFLAGS          = -O3 -w -DDOUBLE_PRECISION
NVCCFLAGS       = -O3 -arch=sm_60 -w -DDOUBLE_PRECISION

INCPATH         = .

OBJECTS		= radiator_cuda.o \
			radiator.o \
			main.o

TARGET		= radiator_cuda_double

all: $(TARGET)

$(TARGET): main.o radiator_cuda.o radiator.o
	$(NVCC) $^ -o $(TARGET) -I$(INCPATH)

radiator_cuda.o: radiator_cuda.cu
	$(NVCC) radiator_cuda.cu -c $(NVCCFLAGS) -I$(INCPATH)

main.o: main.c
	$(CC) main.c -c $(CFLAGS) -I$(INCPATH)

radiator.o: radiator.c
	$(CC) radiator.c -c $(CFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)