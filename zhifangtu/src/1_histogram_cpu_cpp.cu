#include <iostream>
#include <cstdlib>
#include "cuda_error_check.h"

#define NUM_BINS 256
#define DATA_SIZE 1024

void histogram_cpu(const int* data, int* hist, int data_size) {
    // 初始化直方图
    for (int i = 0; i < NUM_BINS; i++) {
        hist[i] = 0;
    }

    // 计算直方图
    for (int i = 0; i < data_size; i++) {
        hist[data[i]]++;
    }
}

int main() {
    // 使用new进行动态内存分配
    int* data = new int[DATA_SIZE];
    int* hist = new int[NUM_BINS];

    // 初始化数据
    for (int i = 0; i < DATA_SIZE; i++) {
        data[i] = rand() % NUM_BINS;
    }

    // 计算直方图
    histogram_cpu(data, hist, DATA_SIZE);

    // 打印直方图结果
    for (int i = 0; i < NUM_BINS; i++) {
        std::cout << "Bin " << i << ": " << hist[i] << std::endl;
    }

    // 使用delete释放内存
    delete[] data;
    delete[] hist;

    return 0;
}
