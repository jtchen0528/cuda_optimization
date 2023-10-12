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

void printArray(float* A, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void initArray(float* A, int M, int N, float val) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // A[i * N + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            A[i * N + j] = val;
        }
    }
}

void initArrayRandom(float* A, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = static_cast <float> (rand() % 10000) / static_cast <float> (10000);
            // A[i * N + j] = i * N + j;
        }
    }
}

void print2DArray(float* A, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void check2DArray(float* A, float *B, int M, int N) {
    bool ans = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(A[i * N + j] - B[i * N + j]) > 0.1f) ans = false;
            assert(ans);
        }
    }
    std::cout << "Correct!" << std::endl;
}


void matmul_serial(float * A, float * B, float * Z, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                Z[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

__global__ void naive(float * A, float * B, float * Z, int m, int n, int p) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[k * p + j];
        }
        Z[i * p + j] = sum;
    }
}

__global__ void naive_transposed(float * A, float * B, float * Z, int m, int n, int p) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[j * n + k];
        }
        Z[i * p + j] = sum;
    }
}

__global__ void transpose(float * A, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp = 0.0f;
    if (i < m && j < n) {
        tmp = A[i * n + j];
    }
    __syncthreads();
    if (i < m && j < n) {
        A[j * m + i] = tmp;
    }
}


__global__ void matmul_tiled(float * A, float * B, float * Z, int m, int n, int p) {
    __shared__ float A_shared [BLOCKDIM][BLOCKDIM];
    __shared__ float B_shared [BLOCKDIM][BLOCKDIM];

    size_t tileIdx = (n + BLOCKDIM - 1) / BLOCKDIM;
    int i, j;
    float sum = 0.0f;
    for (int tid = 0; tid < tileIdx; tid++) {
        j = tid * blockDim.x + threadIdx.x;
        i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < m && j < n) {
            A_shared[threadIdx.y][threadIdx.x] = A[i * n + j];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0;
        }

        j = blockIdx.x * blockDim.x + threadIdx.x;
        i = tid * blockDim.y + threadIdx.y;
        if (i < n && j < p) {
            B_shared[threadIdx.y][threadIdx.x] = B[i * p + j];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCKDIM; k++) {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }
        __syncthreads();
    }
    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < p) {
        Z[i * p + j] = sum;
    }
}

__global__ void matmul_tiled_transposed(float * A, float * B, float * Z, int m, int n, int p) {
    __shared__ float A_shared [BLOCKDIM][BLOCKDIM];
    __shared__ float B_shared [BLOCKDIM][BLOCKDIM];

    size_t tileIdx = (n + BLOCKDIM - 1) / BLOCKDIM;
    int i, j;
    float sum = 0.0f;
    for (int tid = 0; tid < tileIdx; tid++) {
        j = tid * blockDim.x + threadIdx.x;
        i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < m && j < n) {
            A_shared[threadIdx.y][threadIdx.x] = A[i * n + j];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0;
        }

        j = blockIdx.x * blockDim.x + threadIdx.x;
        i = tid * blockDim.y + threadIdx.y;
        if (i < n && j < p) {
            B_shared[threadIdx.x][threadIdx.y] = B[j * n + i];
        } else {
            B_shared[threadIdx.x][threadIdx.y] = 0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCKDIM; k++) {
            sum += A_shared[threadIdx.y][k] * B_shared[threadIdx.x][k];
        }
        __syncthreads();
    }
    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < p) {
        Z[i * p + j] = sum;
    }
}

void execute_transposed_matmul(dim3 gridDim, dim3 blockDim, float * A, float * B, float * Z, int m, int n, int p){
    dim3 gridDimT((p + BLOCKDIM - 1) / BLOCKDIM, (n + BLOCKDIM - 1) / BLOCKDIM, 1);
    transpose<<<gridDimT, blockDim>>>(B, n, p);
    matmul_tiled_transposed<<<gridDim, blockDim>>>(A, B, Z, m, n, p);
}

void execute_naive_transposed_matmul(dim3 gridDim, dim3 blockDim, float * A, float * B, float * Z, int m, int n, int p){
    dim3 gridDimT((p + BLOCKDIM - 1) / BLOCKDIM, (n + BLOCKDIM - 1) / BLOCKDIM, 1);
    transpose<<<gridDimT, blockDim>>>(B, n, p);
    naive_transposed<<<gridDim, blockDim>>>(A, B, Z, m, n, p);
}



int main (int argc, char **argv) {
    printf("CUDA reduciton test\n");

    // uint16_t M = BLOCKDIM * BLOCKDIM;
    // uint16_t N = BLOCKDIM * BLOCKDIM;
    // uint16_t P = BLOCKDIM * BLOCKDIM;

    uint16_t M = BLOCKDIM * 7;
    uint16_t N = BLOCKDIM * 7;
    uint16_t P = BLOCKDIM * 7;

    dim3 blockDim(BLOCKDIM, BLOCKDIM, 1);
    dim3 gridDim((P + BLOCKDIM - 1) / BLOCKDIM, (M + BLOCKDIM - 1) / BLOCKDIM, 1);

    int Asize = M * N;
    int Bsize = N * P;
    int Zsize = M * P;

    float *A = new float[Asize];
    float *B = new float[Bsize];
    float *Z = new float[Zsize];
    float *Z_ref = new float[Zsize];
    float *A_device = NULL, *B_device = NULL, *Z_device = NULL;

    initArrayRandom(A, M, N);
    initArrayRandom(B, N, P);
    // initArray(A, M, N, 1.0f);
    // initArray(B, N, P, 1.0f);
    initArray(Z, M, P, 0.0f);
    initArray(Z_ref, M, P, 0.0f);
    matmul_serial(A, B, Z_ref, M, N, P);

    cudaMalloc(&A_device, sizeof(float) * Asize);
    cudaMalloc(&B_device, sizeof(float) * Bsize);
    cudaMalloc(&Z_device, sizeof(float) * Zsize);

    cudaMemcpy(A_device, A, sizeof(float) * Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, sizeof(float) * Bsize, cudaMemcpyHostToDevice);
    cudaMemcpy(Z_device, Z, sizeof(float) * Zsize, cudaMemcpyHostToDevice);

    // matmul_tiled<<<gridDim, blockDim>>>(A_device, B_device, Z_device, M, N, P);
    execute_transposed_matmul(gridDim, blockDim, A_device, B_device, Z_device, M, N, P);
    cudaDeviceSynchronize();

    cudaMemcpy(Z, Z_device, sizeof(float) * Zsize, cudaMemcpyDeviceToHost);
    print2DArray(Z, 1, P);
    // print2DArray(B, N, P);
    // dim3 gridDimT((P + BLOCKDIM - 1) / BLOCKDIM, (N + BLOCKDIM - 1) / BLOCKDIM, 1);
    // transpose<<<gridDimT, blockDim>>>(B_device, Z_device, N, P);
    // cudaMemcpy(B, B_device, sizeof(float) * Bsize, cudaMemcpyDeviceToHost);
    // // // print2DArray(A, M, N);
    // print2DArray(B, P, N);

    print2DArray(Z_ref, 1, P);
    check2DArray(Z, Z_ref, M, P);



    return 0;
}