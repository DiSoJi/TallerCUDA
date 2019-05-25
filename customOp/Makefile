NVCC = nvcc

all: dotProduct_CUDA

%.o : %.cu
	$(NVCC) -c $< -o $@

matmul_CUDA : dotProduct.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a dotProduct
