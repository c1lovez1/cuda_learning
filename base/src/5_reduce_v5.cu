#include "cuda_error_check.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>

typedef int INT;
typedef float FLOAT;

__device__ void WarpSharedMemReduce(volatile float *smem, int tid)
{
    float x = smem[tid];
    x += smem[tid + 32];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();
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
__global__ void reduce_v0(float *d_in, float *d_out, int N)
{
    __shared__ float smem[blockSize];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockSize * 2 + threadIdx.x;
    float mySum = 0.0f;

    if (gtid < N)
        mySum = d_in[gtid];
    if (gtid + blockSize < N)
        mySum += d_in[gtid + blockSize];

    smem[tid] = mySum;
    __syncthreads();

    if (blockSize >= 1024)
    {
        if (threadIdx.x < 512)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (threadIdx.x < 256)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (threadIdx.x < 64)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        WarpSharedMemReduce(smem, tid);
    }

    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(float *out, float groundtruth, int n)
{
    float res = 0;
    for (int i = 0; i < n; i++)
    {
        res += out[i];
    }
    return fabs(res - groundtruth) < 1e-5;
}

int main()
{
    FLOAT milliseconds = 0;
    int N = 25600000;
    int nbyte = N * sizeof(float);

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 256;
    const int GridSize = (N + blockSize * 2 - 1) / (blockSize * 2);

    float *a = (float *)malloc(nbyte);
    float *d_a;
    cudaMalloc((void **)&d_a, nbyte);

    float *out = (float *)malloc(GridSize * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, GridSize * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
    }

    cudaMemcpy(d_a, a, nbyte, cudaMemcpyHostToDevice);

    float groundtruth = N * 1.0f;

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_v0<blockSize><<<Grid, Block>>>(d_a, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 第二次归约，处理每个块的结果
    float final_result = 0.0f;
    for (int i = 0; i < GridSize; i++)
    {
        final_result += out[i];
    }

    bool is_right = fabs(final_result - groundtruth) < 1e-5;

    if (is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        printf("groundtruth is: %f \n", groundtruth);
        printf("final_result is: %f \n", final_result);
    }
    printf("reduce_v0 latency = %f ms\n", milliseconds);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_out);

    free(a);
    free(out);

    return 0;
}
