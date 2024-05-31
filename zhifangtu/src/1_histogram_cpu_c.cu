#include <stdio.h>
#include <stdlib.h>
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
    int* data = (int*)malloc(DATA_SIZE * sizeof(int));
    int* hist = (int*)malloc(NUM_BINS * sizeof(int));

    if (data == NULL || hist == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 初始化数据
    for (int i = 0; i < DATA_SIZE; i++) {
        data[i] = rand() % NUM_BINS;
    }

    // 计算直方图
    histogram_cpu(data, hist, DATA_SIZE);

    // 打印直方图结果
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d: %d\n", i, hist[i]);
    }

    // 释放内存
    free(data);
    free(hist);

    return 0;
}
