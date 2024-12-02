#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double get_clock() {
    struct timeval tv;
    int ok = gettimeofday(&tv, NULL);
    if (ok < 0) {
        printf("gettimeofday error\n");
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ void gpuHistogram(int* data, int* histogram, int size, int bins, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int bin = (data[idx] * bins) / range;
        atomicAdd(&histogram[bin], 1);
    }
}

int main() {
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int bins = 10, range = 100;

    for (int idx = 0; idx < 5; idx++) {
        int size = sizes[idx];
        int* data = (int*)malloc(size * sizeof(int));
        int* histogram = (int*)malloc(bins * sizeof(int));

        // Initialize data
        for (int i = 0; i < size; i++) {
            data[i] = rand() % range;
        }

        // Allocate device memory
        int *d_data, *d_histogram;
        cudaMalloc(&d_data, size * sizeof(int));
        cudaMalloc(&d_histogram, bins * sizeof(int));
        cudaMemset(d_histogram, 0, bins * sizeof(int));

        cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

        // Kernel configuration
        int threads = 1024;
        int blocks = (size + threads - 1) / threads;

        double t0 = get_clock();
        gpuHistogram<<<blocks, threads>>>(d_data, d_histogram, size, bins, range);
        cudaDeviceSynchronize();
        double t1 = get_clock();

        cudaMemcpy(histogram, d_histogram, bins * sizeof(int), cudaMemcpyDeviceToHost);

        printf("GPU Histogram (Non-Strided) - Size %d: %f s\n", size, t1 - t0);

        free(data);
        free(histogram);
        cudaFree(d_data);
        cudaFree(d_histogram);
    }

    return 0;
}
