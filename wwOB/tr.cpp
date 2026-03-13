#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cblas.h>

void simple_trsm(int upper, int n, int m, const double *A, double *B) {
    if (upper) {
        for (int j = 0; j < m; j++) {
            for (int i = n - 1; i >= 0; i--) {
                double sum = B[i * m + j];
                for (int k = i + 1; k < n; k++) {
                    sum -= A[i * n + k] * B[k * m + j];
                }
                B[i * m + j] = sum / A[i * n + i];
            }
        }
    } else {
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                double sum = B[i * m + j];
                for (int k = 0; k < i; k++) {
                    sum -= A[i * n + k] * B[k * m + j];
                }
                B[i * m + j] = sum / A[i * n + i];
            }
        }
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

double geometric_mean(double *arr, int n) {
    double s = 0.0;

    for (int i = 0; i < n; i++) {
        s += log(arr[i]);
    }

    return exp(s / n);
}

void fill_lower(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) A[i * n + j] = 0.0;
            else A[i * n + j] = 1.0 + rand() % 5;
        }
        A[i * n + i] += n;
    }
}

void fill_upper(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) A[i * n + j] = 0.0;
            else A[i * n + j] = 1.0 + rand() % 5;
        }
        A[i * n + i] += n;
    }
}

void fill_B(double *B, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        B[i] = 1.0 + rand() % 10;
    }
}

int main() {

    int n = 2300;
    int m = 5000;
    int upper = 0;
    int runs = 10;

    srand(0);

    double *A = new double[n * n];
    double *B = new double[n * m];
    double *B_copy = new double[n * m];

    if (upper) fill_upper(A, n);
    else fill_lower(A, n);

    fill_B(B, n, m);

    int threads_list[5] = {1, 2, 4, 8, 16};

    for (int t = 0; t < 5; t++) {

        int threads = threads_list[t];

        char cmd[32];
        sprintf(cmd, "OPENBLAS_NUM_THREADS=%d", threads);
        putenv(cmd);

        double perf[runs];
        double my_times[runs];
        double blas_times[runs];

        printf("\nthreads = %d\n", threads);

        for (int r = 0; r < runs; r++) {

            memcpy(B_copy, B, sizeof(double) * n * m);

            double t1 = get_time();
            simple_trsm(upper, n, m, A, B_copy);
            double t2 = get_time();

            my_times[r] = t2 - t1;

            memcpy(B_copy, B, sizeof(double) * n * m);

            t1 = get_time();
            cblas_dtrsm(
                CblasRowMajor,
                CblasLeft,
                upper ? CblasUpper : CblasLower,
                CblasNoTrans,
                CblasNonUnit,
                n,
                m,
                1.0,
                A,
                n,
                B_copy,
                m
            );
            t2 = get_time();

            blas_times[r] = t2 - t1;

            perf[r] = (blas_times[r] / my_times[r]) * 100.0;

            printf("run %d: my = %f sec, openblas = %f sec, performance = %f %%\n",
                   r + 1, my_times[r], blas_times[r], perf[r]);
        }

        double gm = geometric_mean(perf, runs);

        printf("geometric mean performance = %f %%\n", gm);
    }

    delete[] A;
    delete[] B;
    delete[] B_copy;

    return 0;
}