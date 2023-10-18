#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>

const int WARPSIZE = 32;
const int WARPSIZE_LOG2 = 5;
const int BLOCKDIM = 32;
const int BLOCKSIZE = BLOCKDIM * BLOCKDIM;
const int BINSIZE = 1 << 7;

void printArray(int* A, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void check1DArray(int* A, int *B, int N) {
    for (int i = 0; i < N; i++) {
        assert(A[i] == B[i]);
    }
    std::cout << "Correct!" << std::endl;
}


void histogram_serial(int *A, int *Z, int N) {
    for (int i = 0; i < N; i++) {
        Z[A[i]]++;
    }
}

__global__ void histogram_naive (int *A, int *Z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = A[idx];
    atomicAdd(&Z[val], 1);
}

__global__ void histogram_sm (int *A, int *Z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    __shared__ int smBin [BINSIZE];
    if (tidx < BINSIZE) 
        smBin[tidx] = 0;
    __syncthreads();

    atomicAdd(smBin + A[idx], 1);
    __syncthreads();

    if (tidx < BINSIZE) {
        atomicAdd(&Z[tidx], smBin[tidx]);
        // Z[tidx] = smBin[tidx];
    }
}


int main (int argc, char **argv) {
    printf("CUDA histogram test\n");

    int N = BLOCKSIZE * BLOCKSIZE;

    int blocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;

    int *input = new int[N];
    int *output = new int[BINSIZE];
    int *output_ref = new int[BINSIZE];
    int *input_device = NULL, *output_device = NULL;

    for (int i = 0; i < N; i++) {
        input[i] = i % BINSIZE;
    }

    for (int i = 0; i < BINSIZE; i++) {
        output[i] = 0;
        output_ref[i] = 0;
    }
    
    histogram_serial(input, output_ref, N);
    // printArray(output_ref, BINSIZE);

    cudaMalloc(&input_device, sizeof(float) * N);
    cudaMalloc(&output_device, sizeof(float) * BINSIZE);

    cudaMemcpy(input_device, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, sizeof(float) * BINSIZE, cudaMemcpyHostToDevice);

    histogram_sm<<<blocks, BLOCKSIZE>>>(input_device, output_device, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_device, sizeof(float) * BINSIZE, cudaMemcpyDeviceToHost);
    // cudaMemcpy(input, input_device, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // printArray(output, BINSIZE);

    check1DArray(output, output_ref, BINSIZE);

    return 0;
}
