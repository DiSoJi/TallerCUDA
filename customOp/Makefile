NVCC = nvcc

all: dotProduct_CUDA

matmul_CUDA : dotProduct_CUDA
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a dotProduct_CUDA
