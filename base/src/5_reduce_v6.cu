#include "cuda_error_check.h"
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define WarpSize 32

template <int blockSize>
__device__ float WarpShullfe(float sum)
{
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

// __device__ float WarpShuffle(float sum)
// {
//     sum += __shfl_down_sync(0xffffffff, sum, 16);
//     sum += __shfl_down_sync(0xffffffff, sum, 8);
//     sum += __shfl_down_sync(0xffffffff, sum, 4);
//     sum += __shfl_down_sync(0xffffffff, sum, 2);
//     sum += __shfl_down_sync(0xffffffff, sum, 1);
//     return sum;
// }

__device__ void WarpSharedMemReduce(volatile float *smem, int tid)
{
    float x = smem[tid];
    if (blockDim.x >= 64)
    {
        x += smem[tid + 32];
        __syncwarp();
        smem[tid] = x;
        __syncwarp();
    }
    x += smem[tid + 16];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();
    x += smem[tid + 8];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();
    x += smem[tid + 4];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();
    x += smem[tid + 2];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();
    x += smem[tid + 1];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();
}

template <int blockSize>
__global__ void reduce_warp_level(float *d_in, float *d_out, unsigned int N)
{
    float sum = 0.0f;
    __shared__ float smem[blockSize];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_thread_num = blockDim.x * gridDim.x;
    for (int i = gtid; i < N; i += total_thread_num)
    {
        sum += d_in[i];
    }
    smem[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int i = blockSize / 2; i >= 32; i >>= 1)
    {
        if (tid < i)
        {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }

    // if (threadIdx.x < 32)
    // {
    //     volatile float *vshm = smem;
    //     if (blockDim.x >= 64)
    //     {
    //         vshm[threadIdx.x] += vshm[threadIdx.x + 32];
    //     }
    //     vshm[threadIdx.x] += vshm[threadIdx.x + 16];
    //     vshm[threadIdx.x] += vshm[threadIdx.x + 8];
    //     vshm[threadIdx.x] += vshm[threadIdx.x + 4];
    //     vshm[threadIdx.x] += vshm[threadIdx.x + 2];
    //     vshm[threadIdx.x] += vshm[threadIdx.x + 1];
    // }

    // if (tid < 32)
    // {
    //     WarpSharedMemReduce(smem, tid);
    // }

    // smem[tid] = WarpShullfe<blockSize>(smem[tid]);


    if (tid < 32)
    {
        smem[tid] = WarpShullfe<blockSize>(smem[tid]);
    }

    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
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
    const int GridSize = std::min((N + blockSize - 1) / blockSize, DeviceProp.maxGridSize[0]);

    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));
    // cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    float *out = (float *)malloc(1 * sizeof(float));
    float *d_out;
    float *part_out;
    cudaMalloc((void **)&d_out, 1 * sizeof(float));
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