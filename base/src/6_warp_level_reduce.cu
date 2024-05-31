#include "cuda_error_check.h"
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define WarpSize 32

template <int blockSize>
__device__ float WarpShullfe(float sum)
{
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

template <int blockSize>
__global__ void reduce_warp_level(float *d_in, float *d_out, unsigned int N)
{
    float sum = 0;
    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    unsigned int total_thread_nums = blockSize * gridDim.x;

    for (int i = gtid; i < N; i += total_thread_nums)
    {
        sum += d_in[i];
    }

    __shared__ float WarpSums[blockSize / WarpSize];

    const int laneId = tid % WarpSize;
    const int warpId = tid / WarpSize;

    sum = WarpShullfe<blockSize>(sum);

    if (laneId == 0)
    {
        WarpSums[warpId] = sum;
    }

    __syncthreads();

    sum = (tid < blockSize / WarpSize) ? WarpSums[laneId] : 0;

    if (warpId == 0)
    {
        sum = WarpShullfe<blockSize / WarpSize>(sum);
    }

    if (tid == 0)
    {
        d_out[blockIdx.x] = sum;
    }
}

bool CheckResult(float *out, float groundtruth)
{
    return fabs(*out - groundtruth) < 1e-5;
}

int main()
{
    float millisecond = 0;
    int N = 25600000;

    cudaSetDevice(0);
    cudaDeviceProp DeviceProp;
    cudaGetDeviceProperties(&DeviceProp, 0);

    const int blockSize = 256;
    const int GridSize = std::min((N + blockSize - 1 )/ blockSize, DeviceProp.maxGridSize[0]);

    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));
    // cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    float *out = (float *)malloc(1*sizeof(float));
    float *d_out;
    float *part_out;
    cudaMalloc((void **)&d_out, 1*sizeof(float));
    cudaMalloc((void **)&part_out, GridSize * sizeof(float));
    // cudaMemcpy(d_out,out,sizeof(float),cudaMemcpyHostToDevice);
    // cudaMemcpy(part_out,out,GridSize*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    float groundtruth = N * 1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_warp_level<blockSize><<<Grid, Block>>>(d_a, part_out, N);
    reduce_warp_level<blockSize><<<1, Block>>>(part_out, d_out, GridSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);

    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    bool is_right = CheckResult(out, groundtruth);
    if (is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < 1; i++)
        {
            printf("%lf ", out[i]);
        }
        printf("\n");
    }
    printf("reduce_v6 latency = %f ms\n", millisecond);

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(part_out);
    free(a);
    free(out);

    return 0;
}