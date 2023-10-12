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

const uint16_t KERNEL_SIZE = 7;
__constant__ float kernel_cmem[KERNEL_SIZE * KERNEL_SIZE];
const int smm = (BLOCKDIM + (KERNEL_SIZE / 2) * 2);
const int smn = (BLOCKDIM + (KERNEL_SIZE / 2) * 2);


void printArray(float *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void initArray(float *A, int M, int N, float val)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // A[i * N + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            A[i * N + j] = val;
        }
    }
}

void initArrayRandom(float *A, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = static_cast<float>(rand() % 10000) / static_cast<float>(10000);
            // A[i * N + j] = i * N + j;
        }
    }
}

void print2DArray(float *A, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void check2DArray(float *A, float *B, int M, int N)
{
    bool ans = true;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (fabs(A[i * N + j] - B[i * N + j]) > 0.1f) {
                std::cout << i << " " << j << A[i * N + j] << " " << B[i * N + j] << std::endl;
                ans = false;
            }
            assert(ans);
        }
    }
    std::cout << "Correct!" << std::endl;
}

void conv_serial(float *A, float *K, float *Z, int m, int n, int kernel_size)
{
    int half_kernel = kernel_size / 2;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            int s_i = i - half_kernel, s_j = j - half_kernel;
            for (int a = 0; a < kernel_size; a++) {
                for (int b = 0; b < kernel_size; b++) {
                    int na = s_i + a, nb = s_j + b;
                    if (na >= 0 && na < m && nb >= 0 && nb < n) sum += A[na * n + nb] * K[a * kernel_size + b];
                }
            }
            Z[i * n + j] = sum;
        }
    }
}

__global__ void conv_naive(float *A, float *K, float *Z, int m, int n, int kernel_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int half_kernel = kernel_size / 2;

    float sum = 0.0f;
    int s_i = i - half_kernel, s_j = j - half_kernel;
    for (int a = 0; a < kernel_size; a++) {
        for (int b = 0; b < kernel_size; b++) {
            int na = s_i + a, nb = s_j + b;
            if (na >= 0 && na < m && nb >= 0 && nb < n) sum += A[na * n + nb] * K[a * kernel_size + b];
        }
    }
    Z[i * n + j] = sum;
}

__global__ void conv_cmem_kernel(float *A, float *Z, int m, int n, int kernel_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int half_kernel = kernel_size / 2;

    float sum = 0.0f;
    int s_i = i - half_kernel, s_j = j - half_kernel;
    for (int a = 0; a < kernel_size; a++) {
        for (int b = 0; b < kernel_size; b++) {
            int na = s_i + a, nb = s_j + b;
            if (na >= 0 && na < m && nb >= 0 && nb < n) sum += A[na * n + nb] * kernel_cmem[a * kernel_size + b];
        }
    }
    Z[i * n + j] = sum;
}

__global__ void conv_smem(float *A, float *K, float *Z, int m, int n, int kernel_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int tidi = threadIdx.x, tidj = threadIdx.y;
    int thread_1d_idx = tidi * n + tidj;

    int half_kernel = kernel_size / 2;

    __shared__ float sA [smm][smn];

    // tile value
    sA[(tidi + half_kernel)][(tidj + half_kernel)] = A[i * n + j];
    if (tidi < half_kernel) {
        if (i - half_kernel >= 0)
            sA[tidi][tidj + half_kernel] = A[(i - half_kernel) * n + j];
        else 
            sA[tidi][tidj + half_kernel] = 0.0;
        if (i + blockDim.x < m)
            sA[tidi + blockDim.x + half_kernel][tidj + half_kernel] = A[(i + blockDim.x) * n + j];
        else
            sA[tidi + blockDim.x + half_kernel][tidj + half_kernel] = 0.0;
    }

    if (tidj < half_kernel) {
        if (j - half_kernel >= 0)
            sA[tidi + half_kernel][tidj] = A[i * n + (j - half_kernel)];
        else
            sA[tidi + half_kernel][tidj] = 0.0;
        if (j + blockDim.y < n)
            sA[tidi + half_kernel][tidj + blockDim.y + half_kernel] = A[i * n + j + blockDim.y];
        else
            sA[tidi + half_kernel][tidj + blockDim.y + half_kernel]  = 0.0;
    }

    if (tidi < half_kernel && tidj < half_kernel) {
        if (i - half_kernel >= 0 && j - half_kernel >= 0)
            sA[tidi][tidj] = A[(i - half_kernel) * n + (j - half_kernel)];
        else
            sA[tidi][tidj] = 0.0f;
        if (i - half_kernel >= 0 && j + blockDim.y < n)
            sA[tidi][tidj + half_kernel + blockDim.y] = A[(i - half_kernel) * n + j + blockDim.y];
        else
            sA[tidi][tidj + half_kernel + blockDim.y] = 0.0f;
        if (i + blockDim.x < m && j - half_kernel >= 0)
            sA[tidi + blockDim.x + half_kernel][tidj] = A[(i + blockDim.x) * n + (j - half_kernel)];
        else
            sA[tidi + blockDim.x + half_kernel][tidj] = 0.0f;
        if (i + blockDim.x < m && j + blockDim.y < n)
            sA[tidi + blockDim.x + half_kernel][tidj + half_kernel + blockDim.y] = A[(i + blockDim.x) * n + j + blockDim.y];
        else
            sA[tidi + blockDim.x + half_kernel][tidj + half_kernel + blockDim.y] = 0.0f;
    }

    __syncthreads();

    float sum = 0.0f;
    int s_i = tidi, s_j = tidj;
    for (int a = 0; a < kernel_size; a++) {
        for (int b = 0; b < kernel_size; b++) {
            int na = s_i + a, nb = s_j + b;
            if (na >= 0 && na < m && nb >= 0 && nb < n) sum += sA[na][nb] * K[a * kernel_size + b];
        }
    }
    Z[i * n + j] = sum;
    // Z[i * n + j] = sA[tidi][tidj];
}


int main(int argc, char **argv)
{
    printf("CUDA reduciton test\n");

    // uint16_t M = BLOCKDIM * BLOCKDIM;
    // uint16_t N = BLOCKDIM * BLOCKDIM;
    // uint16_t P = BLOCKDIM * BLOCKDIM;

    uint16_t M = BLOCKDIM * BLOCKDIM;
    uint16_t N = BLOCKDIM * BLOCKDIM;
    uint16_t K = KERNEL_SIZE;

    dim3 blockDim(BLOCKDIM, BLOCKDIM, 1);
    dim3 gridDim((M + BLOCKDIM - 1) / BLOCKDIM, (N + BLOCKDIM - 1) / BLOCKDIM, 1);

    int Asize = M * N;
    int Ksize = K * K;
    int Zsize = M * N;

    float *A = new float[Asize];
    float *Kernel = new float[Ksize];
    float *Z = new float[Zsize];
    float *Z_ref = new float[Zsize];
    float *A_device = NULL, *Kernel_device = NULL, *Z_device = NULL;

    // initArrayRandom(A, M, N);
    // initArrayRandom(B, N, P);
    initArray(A, M, N, 1.0f);
    initArray(Kernel, K, K, 1.0f);
    initArray(Z, M, N, 0.0f);
    initArray(Z_ref, M, N, 0.0f);
    conv_serial(A, Kernel, Z_ref, M, N, K);
    // print2DArray(Z, M, N);

    cudaMalloc(&A_device, sizeof(float) * Asize);
    cudaMalloc(&Kernel_device, sizeof(float) * Ksize);
    cudaMalloc(&Z_device, sizeof(float) * Zsize);

    cudaMemcpy(A_device, A, sizeof(float) * Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(Kernel_device, Kernel, sizeof(float) * Ksize, cudaMemcpyHostToDevice);
    cudaMemcpy(Z_device, Z, sizeof(float) * Zsize, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(kernel_cmem, Kernel, sizeof(float) * Ksize);

    conv_smem<<<gridDim, blockDim>>>(A_device, Kernel_device, Z_device, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(Z, Z_device, sizeof(float) * Zsize, cudaMemcpyDeviceToHost);
    // cudaMemcpy(Kernel, kernel_cmem, sizeof(float) * Ksize, cudaMemcpyDeviceToHost);

    // print2DArray(Kernel, K, K);
    // print2DArray(Z, 1, N);
    // print2DArray(Z_ref, 1, N);
    check2DArray(Z, Z_ref, M, N);

    return 0;
}