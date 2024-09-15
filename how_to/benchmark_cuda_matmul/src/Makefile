# Makefile
CC = nvcc
CFLAGS = -I./include

SOURCES = main.cu
KERNELS = kernels/mm_basic.cu kernels/mm_shared_memory.cu kernels/mm_tiled.cu
OBJECTS = $(SOURCES:.cu=.o) $(KERNELS:.cu=.o)

all: benchmark

benchmark: $(OBJECTS)
	$(CC) $(CFLAGS) -o benchmark $(OBJECTS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) benchmark
