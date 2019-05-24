#include "stdio.h"
#include "stdlib.h"
#include <cuda.h>

#define SIZE 4

__global__ 
void matrixMultiplicationKernel(int* A, int* B, int* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    int tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}
/*
void cudaCheckError() {
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
      exit(0); 
    }
   }
   */
int main (){
    
    //I'm using vectors with 16 elements to represent the matrix (4 rows with 4 values)
    int *first = (int*) malloc(SIZE * SIZE * sizeof(int)); 
    int *second = (int*) malloc(SIZE * SIZE * sizeof(int));
    int *result = (int*) malloc(SIZE * SIZE * sizeof(int));

    //Fill local data:
    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            //i+j gives the position (columb) in the row, for each row.
            first[i+j] = i; 
            second[i+j] = j;
        }
    }

    int *first_gpu;
    int *second_gpu;
    int *result_gpu;

    //Iterations
    int N=SIZE; //size of vector
    //Number of blocks
    int nBytes = N*N*sizeof(int);
    //Block size and number
    int block_size, block_no;
    block_size = 250; //threads per block
    block_no = N/block_size;

    printf("Allocating device memory on host..\n");
    //GPU memory allocation
    cudaMalloc((void **) &first_gpu,  nBytes);
    cudaMalloc((void **) &second_gpu, nBytes);
    cudaMalloc((void **) &result_gpu, nBytes);

    //Work definition////////////////////
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(block_no, 1, 1);
    /////////////////////////////////////
    printf("Copying to device..\n");
    cudaMemcpy(first_gpu, first, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(second_gpu, second, nBytes, cudaMemcpyHostToDevice);

    clock_t start_d=clock();
    printf("Doing GPU Matrix Multiplication\n");
    matrixMultiplicationKernel<<<block_no,block_size>>>(first_gpu, second_gpu, result_gpu, N);
    //cudaCheckError();
    clock_t end_d = clock();
    //Wait for kernel call to finish
    cudaThreadSynchronize();
    
    //Copying data back to host, this is a blocking call and will not start until all kernels are finished
    cudaMemcpy(result, result_gpu, nBytes, cudaMemcpyDeviceToHost);
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    printf("Time it took on GPU: %f", time_d);
    //Free GPU memory
    cudaFree(first_gpu);
    cudaFree(second_gpu);
    cudaFree(result_gpu);
    return 0;
}