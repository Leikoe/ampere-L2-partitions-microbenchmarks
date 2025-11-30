CFLAGS=-arch=sm_80

main: main.cu
	nvcc main.cu -o main $(CFLAGS)

map: map.cu
	nvcc map.cu -o map $(CFLAGS)
