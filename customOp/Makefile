NVCC = nvcc

all: dotProduct_CUDA

%.o : %.cu
	$(NVCC) -c $< -o $@

dotProduct_CUDA : dotProduct_CUDA.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a dotProduct_CUDA
