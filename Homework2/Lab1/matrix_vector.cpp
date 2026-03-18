#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <cstring>


void matrix_vector_product_parallel(double *a, double *b, double *c, int m) {
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < m; j++) {
                a[i * m + j] = i + j;  // Инициализация
            }
            c[i] = 0.0;
        }
        
        // Синхронизация перед вычислениями
        #pragma omp barrier
        
        // Вычисление произведения
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < m; j++) {
                c[i] += a[i * m + j] * b[j];
            }
        }
    }
}


int main() {
    int m = 20000;
    
    double *a = (double*)malloc(sizeof(*a) * m * m);
    double *b = (double*)malloc(sizeof(*b) * m);
    double *c_parallel = (double*)malloc(sizeof(*c_parallel) * m);
    
    // Инициализация вектора
    for (int j = 0; j < m; j++) b[j] = j;
    
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_configs = sizeof(threads) / sizeof(threads[0]);
    
    printf("\n--- Масштабируемость ---\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Time (sec)", "Speedup");
    printf("----------------------------------------\n");
    
    double speedup = 0;

    for (int i = 0; i < num_configs; i++) {
        int num_threads = threads[i];
        omp_set_num_threads(num_threads);
        
        // Обнуляем память
        memset(a, 0, sizeof(*a) * m * m);
        memset(c_parallel, 0, sizeof(*c_parallel) * m);
        
        double t_parallel = omp_get_wtime();
        matrix_vector_product_parallel(a, b, c_parallel, m);
        t_parallel = omp_get_wtime() - t_parallel;
        
        if (i == 0) speedup = t_parallel;
        printf("%-10d %-15.6f %-15.2f\n", num_threads, t_parallel, speedup / t_parallel);
    }
    
    free(a);
    free(b);
    free(c_parallel);
    
    return 0;
}