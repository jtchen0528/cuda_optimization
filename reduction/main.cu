#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

const int WARPSIZE = 32;
const int BLOCKDIM = 32;
const int BLOCKSIZE = BLOCKDIM * BLOCKDIM;

void printArray(float* A, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

__global__ void interleaved(float *input, float *ans) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (index % (stride << 1) == 0) {
            input[index] += input[index + stride];
        }
        __syncthreads();
    }
    if (index == 0) {
        ans[0] = input[index];
    }
}

__global__ void interleaved_2(float *input, float *ans) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            input[index] += input[index + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        ans[0] = input[tid];
    }
}

__global__ void blocked(float *input, float *ans) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            input[tid] += input[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        ans[0] = input[tid];
    }
}

__global__ void shared_memory_blocked(float *input, float *ans) {
    __shared__ float sinput [BLOCKSIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    sinput[threadIdx.x] = input[tid];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            sinput[tid] += sinput[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        ans[0] = sinput[tid];
    }
}

__global__ void half_threads(float *input, float *ans) {
    __shared__ float sinput [BLOCKSIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sinput[tid] = input[index] + input[index + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            sinput[tid] += sinput[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        ans[0] = sinput[tid];
    }
}


__device__ void unroll_last_warp(volatile float *sm) {
    int tid = threadIdx.x;
    sm[tid] += sm[tid + 32];
    if (tid < 16) sm[tid] += sm[tid + 16];
    if (tid < 8) sm[tid] += sm[tid + 8];
    if (tid < 4) sm[tid] += sm[tid + 4];
    if (tid < 2) sm[tid] += sm[tid + 2];
    if (tid < 1) sm[tid] += sm[tid + 1];
}

__global__ void unroll_last(float *input, float *ans) {
    __shared__ float sinput [BLOCKSIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sinput[tid] = input[index] + input[index + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > WARPSIZE; stride >>= 1) {
        if (tid < stride) {
            sinput[tid] += sinput[tid + stride];
        }
        __syncthreads();
    }

    unroll_last_warp(sinput);

    if (tid == 0) {
        ans[0] = sinput[tid];
    }
}

__global__ void unroll_all(float *input, float *ans) {
    __shared__ float sinput [BLOCKSIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sinput[tid] = input[index] + input[index + blockDim.x];
    __syncthreads();

    if (tid < 512) sinput[tid] += sinput[tid + 512];
    __syncthreads();
    if (tid < 256) sinput[tid] += sinput[tid + 256];
    __syncthreads();
    if (tid < 128) sinput[tid] += sinput[tid + 128];
    __syncthreads();
    if (tid < 64) sinput[tid] += sinput[tid + 64];
    __syncthreads();

    unroll_last_warp(sinput);

    if (tid == 0) {
        ans[0] = sinput[tid];
    }
}


__global__ void l1cache_multi_add(float *input, float *ans) {
    __shared__ float sinput [BLOCKSIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * (blockDim.x * 8) + threadIdx.x;
    index <<= 2;
    sinput[tid] = input[index] + input[index + 1] + input[index + 2] + input[index + 3];
    sinput[tid] += input[index + 4 * blockDim.x] + input[index + 4 * blockDim.x + 1] + input[index + 4 * blockDim.x + 2] + input[index + 4 * blockDim.x + 3];
    __syncthreads();

    if (tid < 512) sinput[tid] += sinput[tid + 512];
    __syncthreads();
    if (tid < 256) sinput[tid] += sinput[tid + 256];
    __syncthreads();
    if (tid < 128) sinput[tid] += sinput[tid + 128];
    __syncthreads();
    if (tid < 64) sinput[tid] += sinput[tid + 64];
    __syncthreads();

    unroll_last_warp(sinput);
    // input[threadIdx.x] = sinput[tid];

    if (tid == 0) {
        ans[0] = sinput[tid];
    }
}


__global__ void coalesce_add(float *input, float *ans) {
    __shared__ float sinput [BLOCKSIZE];
    int tid = threadIdx.x;
    unsigned index = blockIdx.x * (blockDim.x * 5) + threadIdx.x;

    sinput[tid] += input[index];
    sinput[tid] += input[index + BLOCKSIZE];
    sinput[tid] += input[index + BLOCKSIZE * 2];
    sinput[tid] += input[index + BLOCKSIZE * 3];
    sinput[tid] += input[index + BLOCKSIZE * 4];
    sinput[tid] += input[index + BLOCKSIZE * 5];
    sinput[tid] += input[index + BLOCKSIZE * 6];
    sinput[tid] += input[index + BLOCKSIZE * 7];
    // sinput[tid] = input[index] + input[index + 1] + input[index + 2] + input[index + 3];
    // sinput[tid] += input[index + 4 * blockDim.x] + input[index + 4 * blockDim.x + 1] + input[index + 4 * blockDim.x + 2] + input[index + 4 * blockDim.x + 3];
    __syncthreads();

    if (tid < 512) sinput[tid] += sinput[tid + 512];
    __syncthreads();
    if (tid < 256) sinput[tid] += sinput[tid + 256];
    __syncthreads();
    if (tid < 128) sinput[tid] += sinput[tid + 128];
    __syncthreads();
    if (tid < 64) sinput[tid] += sinput[tid + 64];
    __syncthreads();

    unroll_last_warp(sinput);

    if (tid == 0) {
        ans[0] = sinput[tid];
    }
}

int main (int argc, char **argv) {
    printf("CUDA reduciton test\n");

    uint16_t N = BLOCKSIZE * 8;

    int blocks = 1;

    float *input = new float[N];
    float *output = new float[N];
    float *ans = new float[1];
    int ans_ref = 0;
    float *input_device = NULL, *ans_device = NULL, *output_device = NULL;

    ans[0] = 0.0f;
    for (uint16_t i = 0; i < N; i++) {
        input[i] = i;
        ans_ref += i;
    }

    cudaMalloc(&input_device, sizeof(float) * N);
    cudaMalloc(&output_device, sizeof(float) * N);
    cudaMalloc(&ans_device, sizeof(float) * 1);

    cudaMemcpy(input_device, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(ans_device, ans, sizeof(float), cudaMemcpyHostToDevice);

    coalesce_add<<<blocks, BLOCKSIZE>>>(input_device, ans_device);
    // shared_memory_blocked<<<blocks, BLOCKSIZE>>>(input_device, ans_device);
    cudaDeviceSynchronize();

    cudaMemcpy(ans, ans_device, sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(input, input_device, sizeof(float) * N, cudaMemcpyDeviceToHost);
    printf("add ans: %f, ans_ref: %d\n", ans[0], ans_ref);
    // printArray(input, BLOCKSIZE);

    assert(ans_ref == ans[0]);

    return 0;
}