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

// GPU Reduction Kernel for Sum
__global__ void gpuReduction(int* input, int* output, int size) {
    __shared__ int partialSum[1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    partialSum[tid] = (gid < size) ? input[gid] : 0;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

int main() {
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    for (int idx = 0; idx < 5; idx++) {
        int size = sizes[idx];
        int* input = (int*)malloc(size * sizeof(int));
        int* output = (int*)malloc(sizeof(int));

        // Initialize array
        for (int i = 0; i < size; i++) {
            input[i] = i + 1;
        }

        // Allocate device memory
        int *d_input, *d_output;
        cudaMalloc(&d_input, size * sizeof(int));
        cudaMalloc(&d_output, sizeof(int));

        cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

        // Kernel configuration
        int threads = 1024;
        int blocks = (size + threads - 1) / threads;

        double t0 = get_clock();
        gpuReduction<<<blocks, threads>>>(d_input, d_output, size);
        cudaDeviceSynchronize();
        double t1 = get_clock();

        cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

        printf("GPU Reduction - Size %d: %f s\n", size, t1 - t0);

        free(input);
        free(output);
        cudaFree(d_input);
        cudaFree(d_output);
    }

    return 0;
}
