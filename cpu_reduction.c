#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Function to get current time in seconds
double get_clock() {
    struct timeval tv;
    int ok = gettimeofday(&tv, NULL);
    if (ok < 0) {
        printf("gettimeofday error\n");
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

void cpuReduction(int* arr, int size, int* sum, int* product, int* min, int* max) {
    *sum = 0;
    *product = 1;
    *min = arr[0];
    *max = arr[0];

    for (int i = 0; i < size; i++) {
        *sum += arr[i];
        *product *= arr[i];
        if (arr[i] < *min) *min = arr[i];
        if (arr[i] > *max) *max = arr[i];
    }
}

int main() {
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    for (int idx = 0; idx < 5; idx++) {
        int size = sizes[idx];
        int* arr = (int*)malloc(size * sizeof(int));
        int sum, product, min, max;

        // Initialize array
        for (int i = 0; i < size; i++) {
            arr[i] = i + 1;
        }

        double t0 = get_clock();
        cpuReduction(arr, size, &sum, &product, &min, &max);
        double t1 = get_clock();

        printf("CPU Reduction - Size %d: %f s\n", size, t1 - t0);

        free(arr);
    }

    return 0;
}
