#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>
#include <cuda_runtime.h>

#define NUM_BINS 256
#define DATA_SIZE 2560000

// CUDA内核函数：使用共享内存加速直方图计算
__global__ void histshared(const unsigned char *d_buffer, const int N, unsigned int *d_histo) {
    // 定义共享内存
    __shared__ unsigned int s_histo[NUM_BINS];

    // 初始化共享内存
    int tid = threadIdx.x;
    if (tid < NUM_BINS) {
        s_histo[tid] = 0;
    }
    __syncthreads();

    // 计算全局索引
    int n = tid + blockDim.x * blockIdx.x;

    // 线程处理数据
    if (n < N) {
        atomicAdd(&s_histo[d_buffer[n]], 1);
    }
    __syncthreads();

    // 将共享内存结果累加到全局内存
    if (tid < NUM_BINS) {
        atomicAdd(&d_histo[tid], s_histo[tid]);
    }
}

bool CheckResult(thrust::host_vector<unsigned int> &out, int* groudtruth, int N) {
    for (int i = 0; i < N; i++) {
        if (out[i] != groudtruth[i]) {
            printf("in checkres, out[i]=%d, gt[i]=%d\n", out[i], groudtruth[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // 初始化主机数据
    thrust::host_vector<unsigned char> h_data(DATA_SIZE);
    for (int i = 0; i < DATA_SIZE; i++) {
        h_data[i] = i % NUM_BINS;
    }

    std::cout<<"1"<<std::endl;

    // 分配和初始化设备向量
    thrust::device_vector<unsigned char> d_data = h_data;
    thrust::device_vector<unsigned int> d_hist(NUM_BINS, 0);

    std::cout<<"2"<<std::endl;

    // 设置内核参数
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (DATA_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    std::cout<<"3"<<std::endl;

    // 创建CUDA事件对象
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout<<"4"<<std::endl;

    // 记录开始时间
    cudaEventRecord(start);

    // 启动内核
    histshared<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_data.data()), 
        DATA_SIZE, 
        thrust::raw_pointer_cast(d_hist.data())
    );

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算内核执行时间
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 将结果从设备传输回主机
    thrust::host_vector<unsigned int> h_hist = d_hist;

    int *groudtruth = (int *)malloc(NUM_BINS * sizeof(int));
    for (int j = 0; j < NUM_BINS; j++) {
        groudtruth[j] = DATA_SIZE / NUM_BINS;
    }

    bool is_right = CheckResult(h_hist, groudtruth, NUM_BINS);
    if (is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for (int i = 0; i < NUM_BINS; i++) {
            printf("%d ", h_hist[i]);
        }
        printf("\n");
    }
    printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);

    // 打印内核执行时间
    printf("histogram_shared_gpu = %f ms\n", milliseconds);

    // 清理CUDA事件对象
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放内存
    free(groudtruth);

    return 0;
}
