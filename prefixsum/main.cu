#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

const int WARPSIZE = 32;
const int WARPSIZE_LOG2 = 5;
const int BLOCKDIM = 32;
const int BLOCKSIZE = BLOCKDIM * BLOCKDIM;

void printArray(float* A, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void check1DArray(float* A, float *B, int N) {
    bool ans = true;
    for (int i = 0; i < N; i++) {
        if (fabs(A[i] - B[i]) > 0.1f) ans = false;
        assert(ans);
    }
    std::cout << "Correct!" << std::endl;
}


void exclusivescan_serial(float *A, float *Z, int N) {
    for (int i = 1; i < N; i++) {
        Z[i] = A[i - 1] + Z[i - 1];
    }
}
//  1 1 1 1 1 1 1 1
//  1 2 1 2 1 2 1 2
//  1 2 1 4 1 2 1 8
//  1 2 1 4 1 2 1 0
//  1 2 1 0 1 2 1 4
//  1 0 1 2 1 4 1 6
//  0 1 2 3 4 5 6 7 

__global__ void prefixsum_naive(float *A, float *Z, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int stride = 1; stride < N; stride <<= 1) {
        if (idx % (stride * 2) == 2 * stride - 1) {
            A[idx] += A[idx - stride];
        }
        __syncthreads();
    }

    if (idx == N - 1) A[idx] = 0;
    __syncthreads();

    for (int stride = N / 2; stride >= 1; stride >>= 1) {
        int stride2 = stride * 2;
        if (idx % stride2 == stride2 - 1) {
            int half = idx - stride;
            float t = A[half];
            A[half] = A[idx];
            A[idx] += t;
        }
        __syncthreads();
    }
    Z[idx] = A[idx];
}

__global__ void prefixsum_naive_sm(float *A, float *Z, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;
    __shared__ float smA [BLOCKSIZE]; 

    smA[tidx] = A[idx];
    __syncthreads();

    for (int stride = 1; stride < BLOCKSIZE; stride <<= 1) {
        if (tidx % (stride * 2) == 2 * stride - 1) {
            smA[tidx] += smA[tidx - stride];
        }
        __syncthreads();
    }

    if (tidx == BLOCKSIZE - 1) smA[tidx] = 0;
    __syncthreads();

    for (int stride = BLOCKSIZE / 2; stride >= 1; stride >>= 1) {
        int stride2 = stride * 2;
        if (tidx % stride2 == stride2 - 1) {
            int half = tidx - stride;
            float t = smA[half];
            smA[half] = smA[tidx];
            smA[tidx] += t;
        }
        __syncthreads();
    }
    Z[idx] = smA[tidx];
}

// 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 
//                 1 2 2 2 2 2 2 2                 1 2 2 2 2 2 2 2
//                 1 2 3 4 4 4 4 4                 1 2 3 4 4 4 4 4
//                 1 2 3 4 5 6 7 8                 1 2 3 4 5 6 7 8
__device__ float warp_inclusive_scan (int tid, float idata, volatile float *scratch) {
    int pos = tid * 2 - tid % WARPSIZE;
    scratch[pos] = 0;
    pos += WARPSIZE;
    scratch[pos] = idata;

    for (int stride = 1; stride < WARPSIZE; stride <<= 1) {
        scratch[pos] += scratch[pos - stride];
    }
    return scratch[pos];
}

__device__ float warp_exclusive_scan (int tid, float idata, volatile float *scratch) {
    return warp_inclusive_scan(tid, idata, scratch) - idata;
}


__global__ void prefixsum_warp(float *A, float *Z, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;
    __shared__ float smA [BLOCKSIZE]; 
    __shared__ float smZ [BLOCKSIZE]; 
    __shared__ float scratch [2 * BLOCKSIZE];

    smA[tidx] = A[idx];
    __syncthreads();

    float idata = smA[tidx];
    //                 1 2 3 4 5 6 7 8                 1 2 3 4 5 6 7 8
    float warpResult = warp_inclusive_scan(tidx, idata, scratch);
    __syncthreads();

    // 8 8 8 8 8 8 8 8 8
    if (tidx % WARPSIZE == WARPSIZE - 1) {
        scratch[tidx >> WARPSIZE_LOG2] = warpResult;
    }

    __syncthreads();

    // 0 8 16 24 32 40 48 56
    if (tidx < (BLOCKSIZE / WARPSIZE)) {
        float val = scratch[tidx];
        scratch[tidx] = warp_exclusive_scan(tidx, val, scratch);
    }
    __syncthreads();

    //               0                      8 16 24 32 40 48 56
    // 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
    smZ[tidx] = scratch[tidx >> WARPSIZE_LOG2] + warpResult - idata;
    Z[idx] = smZ[tidx];
}

int main (int argc, char **argv) {
    printf("CUDA exclusive scan test\n");

    uint16_t N = BLOCKSIZE;

    int blocks = 1;

    float *input = new float[N];
    float *output = new float[N];
    float *output_ref = new float[N];
    float *input_device = NULL, *output_device = NULL;

    for (uint16_t i = 0; i < N; i++) {
        input[i] = 1.0f;
        output[i] = 0.0f;
        output_ref[i] = 0.0f;
    }
    
    exclusivescan_serial(input, output_ref, N);
    // printArray(output_ref, N);

    cudaMalloc(&input_device, sizeof(float) * N);
    cudaMalloc(&output_device, sizeof(float) * N);

    cudaMemcpy(input_device, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, sizeof(float) * N, cudaMemcpyHostToDevice);

    prefixsum_naive_sm<<<blocks, BLOCKSIZE>>>(input_device, output_device, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_device, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // printArray(output, N);

    check1DArray(output, output_ref, N);

    return 0;
}