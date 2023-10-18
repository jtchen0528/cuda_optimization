#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cassert>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
#include "bitonic_sort.h"

using namespace cooperative_groups;

const int WARPSIZE = 32;
const int WARPSIZE_LOG2 = 5;
const int BLOCKDIM = 16;
const int BLOCKSIZE = BLOCKDIM * BLOCKDIM;

void printArray(float* A, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void printArrayInt(int* A, int N) {
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

// 0 1 2 3 4 5 6 7
// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

// sh  0100   
// pos 1001   
//     1001    0000


void sort_serial(float *&A, float *&Z, int N) {
    int stride = 1;
    while (stride < N) {
        for (int i = 0; i < N; i += 2 * stride) {
            int li = 0, ri = 0;
            int lh = i, rh = i + stride, end = i + 2 * stride;
            // std::cout << lh << li << rh << ri << std::endl;
            for (int pos = 0; pos < 2 * stride; pos++) {
                if (lh + li < rh && rh + ri < end) {
                    if (A[lh + li] <= A[rh + ri]) {
                        Z[lh + pos] = A[lh + li];
                        li++;
                    } else {
                        Z[lh + pos] = A[rh + ri];
                        ri++;
                    }
                } else if (lh + li == rh) {
                    Z[lh + pos] = A[rh + ri];
                    ri++;
                } else {
                    Z[lh + pos] = A[lh + li];
                    li++;
                }
            }
        }
        stride <<= 1;
        
        if (stride < N) {
            std::swap(A, Z);
            // std::cout << stride << N << std::endl;
        }
        // break;
    }
}

__inline__ __device__ void swap_float(float *A, int i, int j) {
    float tmp = A[i];
    A[i] = A[j];
    A[j] = tmp;
}

__inline__ __device__ void swap_float_no_tmp(float *A, int i, int j) {
    A[i] += A[j];           //  a + b,   b
    A[j] = A[i] - A[j];     //  a + b,   a
    A[i] -= A[j];            //  b ,   a
}

__inline__ __device__ void swap_float_volatile(volatile float *A, int i, int j) {
    float tmp = A[i];
    A[i] = A[j];
    A[j] = tmp;
}

__global__ void sort_naive(float *A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = 0; it < N; it++) {
        if ((idx & 1) == (it & 1)) {
            if ((idx + 1 < N) && (A[idx] > A[idx + 1])) {
                swap_float(A, idx, idx + 1);
            }
        }
        __syncthreads();
    }
}

__global__ void sort_naive_multi_block(float *A, int N) {
    grid_group g = this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = 0; it < N; it++) {
        if ((idx & 1) == (it & 1)) {
            if ((idx + 1 < N) && (A[idx] > A[idx + 1])) {
                swap_float(A, idx, idx + 1);
            }
        }
        // __syncthreads();
        // break;
        g.sync();
    }
    // A[idx] = N;
}

__global__ void sort_naive_warp_optimized(float *A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    grid_group g = this_grid();

    // A[idx] = 0;
    for (int it = 0; it < N; it++) {
        int oddeven = it & 1;
        int swap_idx = (idx << 1) + oddeven;
        if (swap_idx + 1 < N && A[swap_idx] > A[swap_idx + 1]) {
            swap_float(A, swap_idx, swap_idx + 1);
            // A[swap_idx] = idx;
        }
        g.sync();
        // break;
    }
}


__global__ void sort_naive_warp_optimized_naive_sm(float *A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float smA [BLOCKSIZE];
    int load_idx = idx << 1;
    smA[load_idx] = A[load_idx];
    smA[load_idx + 1] = A[load_idx + 1];
    __syncthreads();
    // A[idx] = 0;
    for (int it = 0; it < N; it++) {
        int oddeven = it & 1;
        int swap_idx = (idx << 1) + oddeven;
        if (swap_idx + 1 < N && smA[swap_idx] > smA[swap_idx + 1]) {
            swap_float(smA, swap_idx, swap_idx + 1);
            // A[swap_idx] = idx;
        }
        __syncthreads();
        // break;
    }

    A[load_idx] = smA[load_idx];
    A[load_idx + 1] = smA[load_idx + 1];
    __syncthreads();

}

__global__ void sort_naive_warp_optimized_sm(float *A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float smA [BLOCKSIZE];
    float4 tmp;
    if (idx < (blockDim.x >> 1)) {
        int load_idx = idx << 2;
        tmp = make_float4(A[load_idx], A[load_idx + 1], A[load_idx + 2], A[load_idx + 3]);
        smA[load_idx] = tmp.x;
        smA[load_idx + 1] = tmp.y;
        smA[load_idx + 2] = tmp.z;
        smA[load_idx + 3] = tmp.w;
    }
    __syncthreads();
    // A[idx] = 0;
    for (int it = 0; it < N; it++) {
        int oddeven = it & 1;
        int swap_idx = (idx << 1) + oddeven;
        if (swap_idx + 1 < N && smA[swap_idx] > smA[swap_idx + 1]) {
            swap_float(smA, swap_idx, swap_idx + 1);
            // A[swap_idx] = idx;
        }
        __syncthreads();
        // break;
    }

    if (idx < (blockDim.x >> 1)) {
        int load_idx = idx << 2;
        tmp = make_float4(smA[load_idx], smA[load_idx + 1], smA[load_idx + 2], smA[load_idx + 3]);
        A[load_idx] = tmp.x;
        A[load_idx + 1] = tmp.y;
        A[load_idx + 2] = tmp.z;
        A[load_idx + 3] = tmp.w;
    }
    __syncthreads();

}


__global__ void odd_even_merge_sort_naive(float *A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // A[idx] = idx;
    for (int stride = 1; stride < N; stride <<= 1) {
        int curr_stride = stride;
        while (curr_stride >= 1) {
            int curr_stride_2 = curr_stride << 1;
            int curr_stride_4 = curr_stride << 2;
            if ((idx % (curr_stride_4)) < curr_stride) {
                if (A[idx] > A[idx + curr_stride]) {
                    swap_float(A, idx, idx + curr_stride);
                }
            }
            else if ((idx % (curr_stride_4)) >= (curr_stride_4 - curr_stride)) {
                if (A[idx] > A[idx - curr_stride]) {
                    swap_float(A, idx, idx - curr_stride);
                }
            }
            curr_stride >>= 1;
            __syncthreads();
        }
    }

    for (int stride = (N >> 1); stride >= 1; stride >>= 1) {
        if (idx % (stride << 1) < stride) {
            if (A[idx] > A[idx + stride]) {
                swap_float(A, idx, idx + stride);
            }
        }
        __syncthreads();
    }
}

__global__ void odd_even_merge_sort_sm(float *A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    __shared__ float smA [BLOCKSIZE];
    smA[tidx] = A[idx];
    __syncthreads();

    // A[idx] = 0;
    for (int stride = 1; stride < N; stride <<= 1) {
        int curr_stride = stride;
        while (curr_stride >= 1) {
            int curr_stride_2 = curr_stride << 1;
            int curr_stride_4 = curr_stride << 2;
            if ((tidx % (curr_stride_4)) < curr_stride) {
                if (smA[tidx] > smA[tidx + curr_stride]) {
                    swap_float_no_tmp(smA, tidx, tidx + curr_stride);
                }
            }
            else if ((tidx % (curr_stride_4)) >= (curr_stride_4 - curr_stride)) {
                if (smA[tidx] > smA[tidx - curr_stride]) {
                    swap_float_no_tmp(smA, tidx, tidx - curr_stride);
                }
            }
            curr_stride >>= 1;
            __syncthreads();
        }
    }

    for (int stride = (N >> 1); stride >= 1; stride >>= 1) {
        if (tidx % (stride << 1) < stride) {
            if (smA[tidx] > smA[tidx + stride]) {
                swap_float_no_tmp(smA, tidx, tidx + stride);
            }
        }
        __syncthreads();
    }

    
    A[idx] = smA[tidx];
    __syncthreads();
}

__global__ void odd_even_merge_sort_first_levels(float *A) {
    // A[idx] = 0;
    __shared__ float smA [BLOCKSIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    smA[tidx] = A[idx];
    __syncthreads();

    for (int stride = 1; stride < (BLOCKSIZE >> 1); stride <<= 1) {
        int curr_stride = stride;
        while (curr_stride >= 1) {
            int curr_stride_2 = curr_stride << 1;
            int curr_stride_4 = curr_stride << 2;
            if ((tidx % (curr_stride_4)) < curr_stride) {
                if (smA[tidx] > smA[tidx + curr_stride]) {
                    swap_float_volatile(smA, tidx, tidx + curr_stride);
                }
                // smA[tidx] = 1;
            }
            else if ((tidx % (curr_stride_4)) >= (curr_stride_4 - curr_stride)) {
                if (smA[tidx] > smA[tidx - curr_stride]) {
                    swap_float_volatile(smA, tidx, tidx - curr_stride);
                }
                // smA[tidx] = 0;
            }
            curr_stride >>= 1;
            __syncthreads();
            // break;
        }
    }

    A[idx] = smA[tidx];
}



__global__ void merge_sort_kernel (float *A, float *scratch, int N, int chunksize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_chunksize = chunksize >> 1;
    int lh = idx * chunksize, rh = lh + half_chunksize;
    int li = 0, ri = 0;

    if (lh < N) {
        while (li < half_chunksize && ri < half_chunksize) {
            int pos = lh + li + ri;
            int lv = A[lh + li], rv = A[rh + ri];
            if (lv < rv) {
                scratch[pos] = lv;
                li++;
            } else {
                scratch[pos] = rv;
                ri++;
            }
        }

        while (li < half_chunksize) {
            scratch[lh + li + ri] = A[lh + li];
            li++;
        }

        while (ri < half_chunksize) {
            scratch[lh + li + ri] = A[rh + ri];
            ri++;
        }
    }
    // A[idx] = idx;
}

void merge_sort(float *&A, float *&scratch, int N) {
    int chunksize = 2;
    int chunk = N >> 1;
    while (chunk > 0) {
        int blockcnt = (chunk + BLOCKSIZE - 1) / BLOCKSIZE;
        // std::cout << blockcnt << " " << chunk << " " << chunksize << std::endl;
        merge_sort_kernel<<<blockcnt, BLOCKSIZE>>>(A, scratch, N, chunksize);
        chunk >>= 1;
        chunksize <<= 1;
        cudaMemcpy(A, scratch, sizeof(float) * N, cudaMemcpyDeviceToDevice);
        // break;
    }
}


int main (int argc, char **argv) {
    printf("CUDA exclusive scan test\n");

    int N = BLOCKSIZE * 8;

    int blocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;

    int *input = new int[N];
    float *output = new float[N];
    float *output_ref = new float[N];
    float *input_ref = new float[N];
    int *input_device = NULL, *output_device = NULL;

    for (int i = 0; i < N; i++) {
        input[i] = N - i;
        input_ref[i] = N - i;
        output[i] = 0.0f;
        output_ref[i] = 0.0f;
    }
    
    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "supportsCoopLaunch: " << supportsCoopLaunch << " deviceProp.multiProcessorCount: " << deviceProp.multiProcessorCount << std::endl;

    // printArray(input, N);
    sort_serial(input_ref, output_ref, N);
    // printArray(input, N);
    printArray(output_ref, N);

    cudaMalloc(&input_device, sizeof(float) * N);
    cudaMalloc(&output_device, sizeof(float) * N);

    cudaMemcpy(input_device, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, sizeof(float) * N, cudaMemcpyHostToDevice);

    // initialize, then launch
    // void *kernelArgs[] = { (void *)&input_device, (void *)&N};

    // cudaLaunchCooperativeKernel((void*)sort_naive_multi_block, blocks, BLOCKSIZE, kernelArgs);
    // odd_even_merge_sort_multi_block <<<blocks, BLOCKSIZE>>> (input_device, N);
    // odd_even_merge_sort_multi_block(input_device, N);
    Kernel_driver(input_device, N, blocks, BLOCKSIZE >> 1);
    // merge_sort(input_device, output_device, N);
    cudaDeviceSynchronize();

    cudaMemcpy(input, input_device, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // printArrayInt(input, N);

    check1DArray(output, output_ref, N);

    return 0;
}