#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

const int BLOCKSIZE = 1024;
const int WARPSIZE = 32;

// reduction over 2N elements

// serial operation: O(N) 

// Parallel opeartions:
// CPU / GPU 


// CUDA
// Time O(N * log (N) / N) O(log N) 
// Space O(1)
// N + N/2 + N / 4 ....  1  = Log N
// algorigthm

// stride = 1
// 1 1 1 1 1 1 1 1 1 1 1 1 
// ^ x ^ x ^ x ^ x ^ x ^ x
// 2 1 2 1 2 1 2 1 2 1 2 1   // N
//t0   1   2   3   4...
// tid = 0 -> A[0] A[1]
// tid = 1 -> A[2] A[3]
// A[index * 2] = A[index * 2 + 1];
// 0  0 1
// 1  2 3
// 2  4 5

// stride <<= 1  = 2
// 2 1 2 1 2 1 2 1 2 1 2 1
// ^   x   ^   x   ^   x
// 4   2   4   2   4   2    //  N/2

// stirde == N
// 4   2   4  ....  2   4   2
// sum

// 1. warp divergence
// 2. Memeory Coalescing
// 3. Multiple Loading
// 4. Shared bank conflict SM


// 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 
// 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
// ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^   => N   warp size == 8
// 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1
// ^ ^ ^ ^ x x x x                   => N / 2
// 

//    adding +

template<typename T>
__device__ void unrollwarp(volatile T *sm, int tid) {
    sm[tid] += sm[tid + 32];
    if (tid < 16) sm[tid] += sm[tid + 16];
    if (tid < 8) sm[tid] += sm[tid + 8];
    if (tid < 4) sm[tid] += sm[tid + 4];
    if (tid < 2) sm[tid] += sm[tid + 2];
    if (tid < 1) sm[tid] += sm[tid + 1];
}

__device__ void faddkahan(volatile float *sm, volatile float *c, int a, float fb) {
    float sum = sm[a];
    float t = sum + fb;
    c[a] = t - sum - fb;
    sm[a] = t;
}

template<typename T>
__global__ void sumreduction(T * input, T *ans) {
    __shared__ T sm [BLOCKSIZE];
    __shared__ T csm [BLOCKSIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    // sm[tidx] = input[idx] + input[idx + BLOCKSIZE];
    sm[tidx] = input[idx];
    faddkahan(sm, csm, tidx, input[idx + BLOCKSIZE]);
    __syncthreads();

    //        N/2      N
    // 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1
    // 4 4 4 4
    for (int stride = (BLOCKSIZE >> 1); stride >= 64; stride >>= 1) {
        if (tidx < stride) {
            T c = 0.0f + csm[tidx] + csm[tidx + stride];
            faddkahan(sm, csm, tidx, sm[tidx + stride] + c);
            // sm[tidx] += sm[tidx + stride];
        }
        __syncthreads();
        // if (tidx < stride) {
        //     sm[tidx] += c;
        // }
    }

    unrollwarp(sm, tidx);

    if (tidx == 0) ans[0] = sm[0];
}

template<typename T>
__global__ void set_input(T * input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    input[idx] = (1.0f * pow(2, (-127 + min(idx, 128))));
}

int main () {
    int N = BLOCKSIZE;
    float * input_device = NULL, *ans_device = NULL;
    float *ans = new float [1];
    cudaMalloc(&input_device, sizeof(float) * N * 2);
    cudaMalloc(&ans_device, sizeof(float) * 1);
    int blocks = (N * 2 + BLOCKSIZE - 1) / BLOCKSIZE;
    set_input<float><<<blocks, BLOCKSIZE>>>(input_device);
    sumreduction<float><<<blocks, BLOCKSIZE>>>(input_device, ans_device);
    cudaMemcpy(ans, ans_device, sizeof(float), cudaMemcpyDeviceToHost);

    printf("ans %e\n", ans[0]);

}


//  sum, ele
// ele -= c
// t = sum + ele  // effective or not
// c = (t - sum) - ele     //
//        ele   - ele   0
//        (ele + c) - ele  