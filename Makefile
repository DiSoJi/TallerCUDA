NVCC = nvcc

all: matmul_CUDA

matmul_CUDA : matmul_CUDA
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a matmul_CUDA
