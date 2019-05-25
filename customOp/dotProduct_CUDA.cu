#include "stdio.h"
#include "stdlib.h"
#include <cuda.h>

#define SIZE 100


__global__
void dotProdKernel(int *A, int *B, int *C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    *C += A[i] * B[i];
}

void dotProd_CPU(int *A, int *B, int *C, int N){
    for (int i = 0; int < N; i++){
        *C += A[i] * B[i];
    }
}

int main() {

    int nBytes = SIZE * sizeof(int);
    int *first = (int*) malloc(nBytes); 
    int *second = (int*) malloc(nBytes);
    int *result = (int*) malloc(sizeof(int));
    *result = 0;

    int block_size, block_no;
    block_size = 250; //threads per block
    block_no = SIZE/block_size;

    //Data filling:
    for (int i = 0; i < SIZE; i++){
        first[i] = i;
        second[i] = i;
    }
    int *first_gpu;
    int *second_gpu;
    int *result_gpu;

    printf("Allocating device memory on host..\n");
    //GPU memory allocation
    cudaMalloc((void **) &first_gpu,  nBytes);
    cudaMalloc((void **) &second_gpu, nBytes);
    cudaMalloc((void **) &result_gpu, sizeof(int));

    //Work definition////////////////////
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(block_no, 1, 1);
    /////////////////////////////////////
    printf("Copying to device..\n");
    cudaMemcpy(first_gpu, first, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(second_gpu, second, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(result_gpu, result, sizeof(int), cudaMemcpyHostToDevice);

    clock_t start_d = clock();
    printf("Doing GPU Matrix Multiplication\n");
    dotProdKernel<<<block_no,block_size>>>(first_gpu, second_gpu, result_gpu, SIZE);
    //cudaCheckError();
    clock_t end_d = clock();

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    //Copying data back to host, this is a blocking call and will not start until all kernels are finished
    cudaMemcpy(result, result_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    printf("Time it took on GPU: %f", time_d);

    printf("Doing work on CPU \n");
    clock_t start = clock();
    dotProd_CPU(first,second,result,SIZE);
    clock_t end = clock();
    double time = (double)(end-start)/CLOCKS_PER_SEC;
    printf("Time it took on CPU: %f", time);
    
    //Free GPU memory
    cudaFree(first_gpu);
    cudaFree(second_gpu);
    cudaFree(result_gpu);

    return 0;
}
