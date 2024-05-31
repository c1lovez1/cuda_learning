#include "cuda_error_check.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>
#include <cstring> // Add this header for memset

#define DATA_SIZE 10240 * 10240
#define NUM_BINS 256
#define ThreadsperBlock 512

typedef float DataType;

void histogram(const unsigned char *h_data, DataType *h_hist, int N)
{
    for (int i = 0; i < N; i++)
    {
        h_hist[h_data[i]]++;
    }
}

__global__ void histogram_baseline(const unsigned char *d_data, DataType *d_hist, int N)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        atomicAdd(&d_hist[d_data[idx]], 1.0f);
    }
}

__global__ void histogram_shared(const unsigned char *d_data, DataType *d_hist, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ unsigned int smem[NUM_BINS];

    if (tid < NUM_BINS)
    {
        smem[tid] = 0;
    }
    __syncthreads();

    if (gtid < N)
    {
        atomicAdd(&smem[d_data[gtid]], 1);
        __syncthreads();
        // atomicAdd(&d_hist[tid],smem[tid]);
    }

    if (tid < NUM_BINS)
    {
        atomicAdd(&d_hist[tid], smem[tid]);
    }
}

__global__ void histogram_Parallelism(const unsigned char *d_data, DataType *d_hist, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ unsigned int smem[NUM_BINS];

    if (tid < NUM_BINS)
    {
        smem[tid] = 0;
    }
    __syncthreads();

    const unsigned int offset = blockDim.x * gridDim.x;

#pragma unroll
    for (; gtid < N; gtid += offset)
    {
        atomicAdd(&smem[d_data[gtid]], 1);
        __syncthreads();
        // atomicAdd(&d_hist[tid],smem[tid]);
    }
    // __syncthreads();

    if (tid < NUM_BINS)
    {
        atomicAdd(&d_hist[tid], smem[tid]);
    }
}

bool CheckResult(DataType *h_hist, DataType *cpu_hist, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (h_hist[i] != cpu_hist[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{
    float milliseconds = 0.0f;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // unsigned char *h_data = (unsigned char *)malloc(DATA_SIZE * sizeof(unsigned char));
    unsigned char *h_data = new unsigned char[DATA_SIZE];

    DataType h_hist[NUM_BINS] = {0};
    DataType cpu_hist[NUM_BINS] = {0};

    // memset(h_hist, 0, NUM_BINS * sizeof(DataType));
    // memset(cpu_hist, 0, NUM_BINS * sizeof(DataType));

    for (int i = 0; i < DATA_SIZE; i++)
    {
        h_data[i] = i % NUM_BINS;
    }

    histogram(h_data, cpu_hist, DATA_SIZE);

    const int BlockperGrid = std::min((DATA_SIZE + ThreadsperBlock - 1) / ThreadsperBlock, deviceProp.maxGridSize[0]);

    unsigned char *d_data;
    DataType *d_hist;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, DATA_SIZE * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_hist, NUM_BINS * sizeof(DataType)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, DATA_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_hist, h_hist, NUM_BINS * sizeof(DataType), cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERROR(cudaMemset(d_hist, 0, NUM_BINS * sizeof(DataType)));  // Initialize device histogram array

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // histogram_baseline<<<BlockperGrid, ThreadsperBlock>>>(d_data, d_hist, DATA_SIZE);
    // histogram_shared<<<BlockperGrid, ThreadsperBlock>>>(d_data, d_hist, DATA_SIZE);
    histogram_Parallelism<<<BlockperGrid, ThreadsperBlock>>>(d_data, d_hist, DATA_SIZE);

    // CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(DataType), cudaMemcpyDeviceToHost));

    if (CheckResult(h_hist, cpu_hist, NUM_BINS))
    {
        std::cout << "Results match!" << std::endl;
    }
    else
    {
        std::cout << "Results do not match!" << std::endl;
    }

    printf("histogram_gpu = %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_hist);
    // free(h_data); // Free dynamically allocated memory
    delete[] h_data; // Free dynamically allocated memory

    return 0;
}

