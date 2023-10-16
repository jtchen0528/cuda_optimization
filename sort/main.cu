#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cassert>
#include <algorithm>

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

__global__ 

int main (int argc, char **argv) {
    printf("CUDA exclusive scan test\n");

    uint16_t N = BLOCKSIZE;

    int blocks = 1;

    float *input = new float[N];
    float *output = new float[N];
    float *output_ref = new float[N];
    float *input_ref = new float[N];
    float *input_device = NULL, *output_device = NULL;

    for (uint16_t i = 0; i < N; i++) {
        input[i] = N - i;
        input_ref[i] = N - i;
        output[i] = 0.0f;
        output_ref[i] = 0.0f;
    }
    
    // printArray(input, N);
    sort_serial(input, output_ref, N);
    // printArray(input, N);
    printArray(output_ref, N);

    cudaMalloc(&input_device, sizeof(float) * N);
    cudaMalloc(&output_device, sizeof(float) * N);

    cudaMemcpy(input_device, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, sizeof(float) * N, cudaMemcpyHostToDevice);

    // prefixsum_naive_sm<<<blocks, BLOCKSIZE>>>(input_device, output_device, N);
    // cudaDeviceSynchronize();

    cudaMemcpy(output, output_device, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // printArray(output, N);

    check1DArray(output, output_ref, N);

    return 0;
}