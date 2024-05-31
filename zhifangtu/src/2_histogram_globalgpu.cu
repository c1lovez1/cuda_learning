#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#define NUM_BINS 256
#define DATA_SIZE 1024000

__global__ void histKernel(const unsigned char *d_buffer, const int N, unsigned int *d_histo)
{

    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < N)
    {
        atomicAdd(&d_histo[d_buffer[n]], 1);
    }
}

int main()
{
    // 初始化数据
    unsigned char h_data[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; i++)
    {
        h_data[i] = rand() % NUM_BINS;
    }

    // 分配和初始化主机内存
    unsigned int h_hist[NUM_BINS] = {0};

    // 分配设备内存
    unsigned char *d_data;
    unsigned int *d_hist;
    cudaMalloc((void **)&d_data, DATA_SIZE * sizeof(unsigned char));
    cudaMalloc((void **)&d_hist, NUM_BINS * sizeof(unsigned int));

    // 将数据从主机传输到设备
    cudaMemcpy(d_data, h_data, DATA_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

    // 启动内核
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (DATA_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, DATA_SIZE, d_hist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 将结果从设备传输回主机
    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("2_histogram_globalgpu = %f ms\n", milliseconds);

    // 清理CUDA事件对象
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // 释放设备内存
    cudaFree(d_data);
    cudaFree(d_hist);

    return 0;
}
