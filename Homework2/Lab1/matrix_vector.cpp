#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <cstring>

// Функция для выделения памяти с проверкой
double* xmalloc(size_t size) {
    double* ptr = (double*)malloc(size);
    if (!ptr) {
        printf("Ошибка выделения памяти\n");
        exit(1);
    }
    return ptr;
}

// Последовательная версия
void matrix_vector_product_serial(double *a, double *b, double *c, int m, int n) {
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

// Параллельная версия с инициализацией данных теми же потоками
void matrix_vector_product_parallel(double *a, double *b, double *c, int m, int n) {
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        
        // Параллельная инициализация матрицы и вектора результата
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i + j;  // Инициализация
            }
            c[i] = 0.0;
        }
        
        // Синхронизация перед вычислениями
        #pragma omp barrier
        
        // Вычисление произведения
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++) {
                c[i] += a[i * n + j] * b[j];
            }
        }
    }
}

// Функция для измерения времени
double wtime() {
    return omp_get_wtime();
}

int main(int argc, char **argv) {
    int m = 20000, n = 20000;  // Размер матрицы
    
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    
    printf("Matrix-vector product (c[%d] = a[%d][%d] * b[%d])\n", m, m, n, n);
    printf("Memory used: %.2f GiB\n", 
           ((double)(m * n + m + n) * sizeof(double)) / (1024.0 * 1024.0 * 1024.0));
    
    // Выделение памяти
    double *a = xmalloc(sizeof(*a) * m * n);
    double *b = xmalloc(sizeof(*b) * n);
    double *c_serial = xmalloc(sizeof(*c_serial) * m);
    double *c_parallel = xmalloc(sizeof(*c_parallel) * m);
    
    // Инициализация вектора b (один поток)
    for (int j = 0; j < n; j++) {
        b[j] = j;
    }
    
    // Последовательный вариант
    double t = wtime();
    matrix_vector_product_serial(a, b, c_serial, m, n);
    t = wtime() - t;
    printf("Elapsed time (serial): %.6f sec.\n", t);
    double serial_time = t;
    
    // Параллельный вариант для разного числа потоков
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_configs = sizeof(threads) / sizeof(threads[0]);
    
    printf("\n--- Масштабируемость ---\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Time (sec)", "Speedup");
    printf("----------------------------------------\n");
    
    for (int i = 0; i < num_configs; i++) {
        int num_threads = threads[i];
        omp_set_num_threads(num_threads);
        
        // Обнуляем память (чтобы first-touch policy сработала заново)
        memset(a, 0, sizeof(*a) * m * n);
        memset(c_parallel, 0, sizeof(*c_parallel) * m);
        
        double t_parallel = wtime();
        matrix_vector_product_parallel(a, b, c_parallel, m, n);
        t_parallel = wtime() - t_parallel;
        
        double speedup = serial_time / t_parallel;
        printf("%-10d %-15.6f %-15.2f\n", num_threads, t_parallel, speedup);
    }
    
    free(a);
    free(b);
    free(c_serial);
    free(c_parallel);
    
    return 0;
}