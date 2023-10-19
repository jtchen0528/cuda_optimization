#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cassert>
#include <queue>
#include <algorithm>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>

#include "CycleTimer.h"
#include "bitonic_sort.h"

const int WARPSIZE = 32;
const int WARPSIZE_LOG2 = 5;
const int BLOCKDIM = 32;
const int BLOCKSIZE = BLOCKDIM * BLOCKDIM;

using namespace std;

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

void swap_int (int *A, int i, int j) {
    int tmp = A[i];
    A[i] = A[j];
    A[j] = tmp;
}
//  2th
//  1  2  3  4  5  6  7
//                 lr p
int findKthMin_quickselect(int *A, int k, int end, int start = 0) {
    int pivot = A[end];
    int l = start, r = end - 1;
    while (l <= r) {
        if (A[l] > pivot) {
            swap(A[l], A[r]);
            r--;
        } else {
            l++;
        }
    }
    r++;
    swap(A[r], A[end]);
    if (r == k) return A[r];
    else if (r > k) return findKthMin_quickselect(A, k, r - 1, start);
    else return findKthMin_quickselect(A, k, end, r + 1);
}

int findKthMin_pq(int *A, int k, int N) {
    priority_queue <int> pq;
    for (int i = 0; i < N; i++) {
        if (pq.size() < k) {
            pq.push(A[i]);
        } else {
            if (pq.top() > A[i]) {
                pq.pop();
                pq.push(A[i]);
            }
        }
    }
    return pq.top();
}

int findKthMin_sort(int *A, int k, int N) {
    sort(A, A + N);
    return A[k];
}

int main (int argc, char **argv) {
    printf("CUDA findKmin test\n");

    int N = BLOCKSIZE * BLOCKSIZE;
    int k = BLOCKSIZE * BLOCKSIZE / 2;

    int *input = new int[N];
    int *output = new int[1];
    int *input_device = NULL, *output_device = NULL;

    for (int i = 0; i < N; i++) {
        input[i] = N - i + BLOCKDIM;
    }
    
    random_device rd;
    mt19937 g(rd());

    shuffle(input, input + N, g);

    double kernelStartTime, kernelEndTime, kernelDuration;
    kernelStartTime = CycleTimer::currentSeconds();
    int ans_ref_quickselect = findKthMin_quickselect(input, k - 1, N - 1);
    kernelEndTime = CycleTimer::currentSeconds();
    kernelDuration = kernelEndTime - kernelStartTime;
    printf("ans_ref_quickselect = %d,  %.3f ms\n", ans_ref_quickselect, 1000.f * kernelDuration);

    kernelStartTime = CycleTimer::currentSeconds();
    int ans_ref_pq = findKthMin_pq(input, k, N);
    kernelEndTime = CycleTimer::currentSeconds();
    kernelDuration = kernelEndTime - kernelStartTime;
    printf("ans_ref_pq = %d,  %.3f ms\n", ans_ref_pq, 1000.f * kernelDuration);

    kernelStartTime = CycleTimer::currentSeconds();
    int ans_ref_sort = findKthMin_sort(input, k - 1, N);
    kernelEndTime = CycleTimer::currentSeconds();
    kernelDuration = kernelEndTime - kernelStartTime;
    printf("ans_ref_sort = %d,  %.3f ms\n", ans_ref_sort, 1000.f * kernelDuration);

    cudaMalloc(&input_device, sizeof(int) * N);

    cudaMemcpy(input_device, input, sizeof(int) * N, cudaMemcpyHostToDevice);
    int blocks = (N + BLOCKSIZE * 2 - 1) / BLOCKSIZE * 2;
    kernelStartTime = CycleTimer::currentSeconds();
    Kernel_driver(input_device, N, blocks, BLOCKSIZE);
    cudaDeviceSynchronize();
    kernelEndTime = CycleTimer::currentSeconds();
    // histogram_sm<<<blocks, BLOCKSIZE>>>(input_device, output_device, N);
    cudaMemcpy(output, input_device, sizeof(int) * k, cudaMemcpyDeviceToHost);
    // cudaMemcpy(input, input_device, sizeof(float) * N, cudaMemcpyDeviceToHost);
    int ans_bitonic_sort = output[k - 1];
    kernelDuration = kernelEndTime - kernelStartTime;
    printf("ans_ref_sort = %d,  %.3f ms\n", ans_ref_sort, 1000.f * kernelDuration);
    // printArray(output, BINSIZE);

    // check1DArray(output, output_ref, BINSIZE);

    return 0;
}
