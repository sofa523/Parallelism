#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <barrier>

using namespace std::chrono;

void thread_work(double* a, double* b, double* c, int m, 
                 int start_row, int end_row,
                 std::barrier<>& barrier) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i * m + j] = i + j;
        }
        c[i] = 0.0;
    }
    
    barrier.arrive_and_wait();
    
    for (int i = start_row; i < end_row; ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += a[i * m + j] * b[j];
        }
        c[i] = sum;
    }
}

void matrix_vector_product_parallel(double* a, double* b, double* c, int m, int num_threads) {
    std::vector<std::thread> threads;
    std::barrier sync_point(num_threads);
    
    int rows_per_thread = m / num_threads;
    int remainder = m % num_threads;
    int current_row = 0;
    
    for (int t = 0; t < num_threads; ++t) {
        int start_row = current_row;
        int end_row = current_row + rows_per_thread + (t < remainder ? 1 : 0);
        current_row = end_row;
        threads.emplace_back(thread_work, a, b, c, m, start_row, end_row, std::ref(sync_point));
    }
    
    for (auto& th : threads) th.join();
}


int main() {
    int sizes[] = {20000, 40000};
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};

    for (int m : sizes) {
        printf("\n=== Matrix size: %d x %d ===\n", m, m);

        double* a = (double*)malloc(sizeof(double) * m * m);
        double* b = (double*)malloc(sizeof(double) * m);
        double* c_serial = (double*)malloc(sizeof(double) * m);
        double* c_parallel = (double*)malloc(sizeof(double) * m);

        for (int j = 0; j < m; ++j) b[j] = j;

        auto start = high_resolution_clock::now();
        matrix_vector_product_parallel(a, b, c_serial, m, 1);
        auto end = high_resolution_clock::now();
        double serial_time = duration<double>(end - start).count();

        printf("%-10s %-15s %-15s\n", "Threads", "Time (sec)", "Speedup");
        printf("%-10d %-15.6f %-15.2f\n", 1, serial_time, 1.0);

        for (int i = 1; i < sizeof(threads)/sizeof(threads[0]); ++i) {
            int num_threads = threads[i];
            memset(a, 0, sizeof(double) * m * m);
            memset(c_parallel, 0, sizeof(double) * m);

            auto pstart = high_resolution_clock::now();
            matrix_vector_product_parallel(a, b, c_parallel, m, num_threads);
            auto pend = high_resolution_clock::now();
            double parallel_time = duration<double>(pend - pstart).count();

            printf("%-10d %-15.6f %-15.2f\n", num_threads, parallel_time, serial_time / parallel_time);
        }

        free(a); free(b); free(c_serial); free(c_parallel);
    }
    return 0;
}