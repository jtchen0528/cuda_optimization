#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Refer: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

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
            // A[i * N + j] = static_cast <float> (rand() % 10000) / static_cast <float> (10000);
            A[i * N + j] = i * N + j;
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
            if (fabs(A[i * N + j] - B[i * N + j]) > 0.1f)
                ans = false;
            assert(ans);
        }
    }
    std::cout << "Correct!" << std::endl;
}

void transpose_serial(float *A, float *Z, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Z[j * m + i] = A[i * n + j];
        }
    }
}

__global__ void copy_naive(float *odata, const float *idata)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    odata[y * width + x] = idata[y * width + x];
}

__global__ void copy(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

__global__ void transposeNaive(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
}

__global__ void transposeCoalesced(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transposeCoalescedBankConflict(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char **argv)
{
    printf("CUDA reduciton test\n");

    // uint16_t M = BLOCKDIM * BLOCKDIM;
    // uint16_t N = BLOCKDIM * BLOCKDIM;
    // uint16_t P = BLOCKDIM * BLOCKDIM;

    uint16_t M = TILE_DIM * TILE_DIM;
    uint16_t N = TILE_DIM * TILE_DIM;

    dim3 blockDim(TILE_DIM, BLOCK_ROWS, 1);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM, 1);

    int Asize = M * N;
    int Zsize = N * M;

    float *A = new float[Asize];
    float *Z = new float[Zsize];
    float *Z_ref = new float[Zsize];
    float *A_device = NULL, *Z_device = NULL;

    initArrayRandom(A, M, N);
    // initArray(A, M, N, 1.0f);
    // initArray(B, N, P, 1.0f);
    initArray(Z, N, M, 0.0f);
    initArray(Z_ref, N, M, 0.0f);
    transpose_serial(A, Z_ref, M, N);

    cudaMalloc(&A_device, sizeof(float) * Asize);
    cudaMalloc(&Z_device, sizeof(float) * Zsize);

    cudaMemcpy(A_device, A, sizeof(float) * Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(Z_device, Z, sizeof(float) * Zsize, cudaMemcpyHostToDevice);

    transposeCoalescedBankConflict<<<gridDim, blockDim>>>(Z_device, A_device);
    cudaDeviceSynchronize();

    cudaMemcpy(Z, Z_device, sizeof(float) * Zsize, cudaMemcpyDeviceToHost);

    // print2DArray(A, M, N);
    // print2DArray(Z, M, N);

    // print2DArray(Z_ref, N, M);
    check2DArray(Z_ref, Z, N, M);

    return 0;
}