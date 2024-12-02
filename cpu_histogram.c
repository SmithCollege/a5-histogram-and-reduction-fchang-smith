#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_clock() {
    struct timeval tv;
    int ok = gettimeofday(&tv, NULL);
    if (ok < 0) {
        printf("gettimeofday error\n");
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// CPU Histogram
void cpuHistogram(int* data, int size, int* histogram, int bins, int range) {
    for (int i = 0; i < bins; i++) histogram[i] = 0;

    for (int i = 0; i < size; i++) {
        int bin = (data[i] * bins) / range;
        histogram[bin]++;
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

        double t0 = get_clock();
        cpuHistogram(data, size, histogram, bins, range);
        double t1 = get_clock();

        printf("CPU Histogram - Size %d: %f s\n", size, t1 - t0);

        free(data);
        free(histogram);
    }

    return 0;
}
