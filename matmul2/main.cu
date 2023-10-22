#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Refer: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// ref: https://github.com/siboehm/SGEMM_CUDA

const int WARPSIZE = 32;
const int BLOCKDIM = 128;
const int BLOCKSIZE = BLOCKDIM * BLOCKDIM;
const int TM = 8;  // THREAD DIM
const int SMDIM = 8;

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
            if (fabs(A[i * N + j] - B[i * N + j]) > 0.1f)
                ans = false;
            assert(ans);
        }
    }
    std::cout << "Correct!" << std::endl;
}

void matmul_serial(float *A, float *B, float *Z, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < n; k++)
            {
                Z[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

__global__ void naive(float *A, float *B, float *Z, int m, int n, int p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < m && y < p)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += A[x * n + k] * B[k * p + y];
        }
        Z[x * p + y] = sum;
    }
}

__global__ void naive_coalesce(float *A, float *B, float *Z, int m, int n, int p)
{
    const int x = blockIdx.x * BLOCKDIM + (threadIdx.x / BLOCKDIM);
    const int y = blockIdx.y * BLOCKDIM + (threadIdx.x % BLOCKDIM);
    if (x < m && y < p)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += A[x * n + k] * B[k * p + y];
        }
        Z[x * p + y] = sum;
    }
}

__global__ void tiled(float *A, float *B, float *Z, int m, int n, int p)
{
    const int tidx = threadIdx.x / BLOCKDIM;
    const int tidy = threadIdx.x % BLOCKDIM;

    __shared__ float smA[BLOCKSIZE];
    __shared__ float smB[BLOCKSIZE];

    A += blockIdx.x * BLOCKDIM * n;                         // (x, 0)
    B += blockIdx.y * BLOCKDIM;                             // (0, y)
    Z += blockIdx.x * BLOCKDIM * p + blockIdx.y * BLOCKDIM; // (x, y);

    float sum = 0.0f;
    for (int tile_idx = 0; tile_idx < n; tile_idx += BLOCKDIM)
    {
        smA[tidx * BLOCKDIM + tidy] = A[tidx * n + tidy];
        smB[tidx * BLOCKDIM + tidy] = B[tidx * p + tidy];
        __syncthreads();
        A += BLOCKDIM;
        B += BLOCKDIM * p;

        for (int i = 0; i < BLOCKDIM; i++)
        {
            sum += smA[tidx * BLOCKDIM + i] * smB[i * BLOCKDIM + tidy];
        }
        __syncthreads();
    }
    Z[tidx * p + tidy] = sum;
}


// half gmem access (9/16 Smem access), boost instruction / cycle not bound by mem IO
__global__ void tiled_lesssm(float *A, float *B, float *Z, int m, int n, int p)
{
    A += blockIdx.y * BLOCKDIM * n;                         //  (bidx * blockdim, 0)
    B += blockIdx.x * BLOCKDIM;                             //  (0, bidy * blockdim)
    Z += blockIdx.y * BLOCKDIM * p + blockIdx.x * BLOCKDIM; //  (bidx * blockdim, bidy * blockdim)

    const int tidx = threadIdx.x / BLOCKDIM;
    const int tidy = threadIdx.x % BLOCKDIM;

    __shared__ float smA [BLOCKDIM * SMDIM],  smB [BLOCKDIM * SMDIM];

    const int smAr = threadIdx.x / SMDIM;
    const int smAc = threadIdx.x % SMDIM;
    const int smBr = threadIdx.x / BLOCKDIM;
    const int smBc = threadIdx.x % BLOCKDIM;

    float per_block_results [TM] = {0.0};

    for (int bidx = 0; bidx < n; bidx += BLOCKDIM) {
        for (int tile_idx = 0; tile_idx < BLOCKDIM; tile_idx += SMDIM) {
            smA[smAr * SMDIM + smAc] = A[smAr * n + smAc];
            smB[smBr * BLOCKDIM + smBc] = B[smBr * p + smBc];
            __syncthreads();

            A += SMDIM;
            B += SMDIM * p;

            for (int bridx = 0; bridx < SMDIM; bridx++) {
                float Btmp = smB[bridx * BLOCKDIM + tidy];
                for (int bcidx = 0; bcidx < TM; bcidx++) {
                    per_block_results[bcidx] += smA[(tidx * TM + bcidx) * SMDIM + bridx] * Btmp;
                }
            }
            __syncthreads();
        }
    }

    for (int resIdx = 0; resIdx < TM; resIdx++)
    {
        Z[(tidx * TM + resIdx) * p + tidy] += per_block_results[resIdx];
    }
}

__global__ void sgemm2DWarpTiling(float *A, float *B, float *C, int M, int K, int N) {
  const int BM = BLOCKDIM;
  const int BN = BLOCKDIM;
  const int BK = SMDIM;
  const int TN = TM;
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

    // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] +=
          threadResults[resIdxM * TN + resIdxN];
    }
  }
}

int main(int argc, char **argv)
{
    printf("CUDA reduciton test\n");

    // uint16_t M = BLOCKDIM * BLOCKDIM;
    // uint16_t N = BLOCKDIM * BLOCKDIM;
    // uint16_t P = BLOCKDIM * BLOCKDIM;

    uint16_t M = BLOCKDIM * 8;
    uint16_t N = BLOCKDIM * 8;
    uint16_t P = BLOCKDIM * 8;

    dim3 blockDim((BLOCKDIM * BLOCKDIM) / (TM * TM));
    dim3 gridDim((M + BLOCKDIM - 1) / BLOCKDIM, (P + BLOCKDIM - 1) / BLOCKDIM, 1);

    int Asize = M * N;
    int Bsize = N * P;
    int Zsize = M * P;

    float *A = new float[Asize];
    float *B = new float[Bsize];
    float *Z = new float[Zsize];
    float *Z_ref = new float[Zsize];
    float *A_device = NULL, *B_device = NULL, *Z_device = NULL;

    // initArrayRandom(A, M, N);
    // initArrayRandom(B, N, P);
    initArray(A, M, N, 1.0f);
    initArray(B, N, P, 1.0f);
    initArray(Z, M, P, 0.0f);
    initArray(Z_ref, M, P, 0.0f);
    matmul_serial(A, B, Z_ref, M, N, P);
    // print2DArray(Z_ref, M, P);

    cudaMalloc(&A_device, sizeof(float) * Asize);
    cudaMalloc(&B_device, sizeof(float) * Bsize);
    cudaMalloc(&Z_device, sizeof(float) * Zsize);

    cudaMemcpy(A_device, A, sizeof(float) * Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, sizeof(float) * Bsize, cudaMemcpyHostToDevice);
    cudaMemcpy(Z_device, Z, sizeof(float) * Zsize, cudaMemcpyHostToDevice);

    sgemmVectorize<<<gridDim, blockDim>>>(A_device, B_device, Z_device, M, N, P);
    cudaDeviceSynchronize();

    cudaMemcpy(Z, Z_device, sizeof(float) * Zsize, cudaMemcpyDeviceToHost);
    // print2DArray(Z, M, P);

    check2DArray(Z, Z_ref, M, P);

    return 0;
}
// 1,813,567  1,821,054  1,818,687