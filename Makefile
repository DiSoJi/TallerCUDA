NVCC = nvcc

all: matmul_CUDA dotProduct_CUDA

%.o : %.cu
	$(NVCC) -c $< -o $@

matmul_CUDA : matmul_CUDA.o
	$(NVCC) $^ -o $@

dotProd_CUDA : dotProduct_CUDA.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a matmul_CUDA
	rm -rf *.o *.a dotProduct_CUDA
