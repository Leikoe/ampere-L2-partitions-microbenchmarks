CFLAGS=-arch=sm_80

main: main.cu
	nvcc main.cu -o main $(CFLAGS)
