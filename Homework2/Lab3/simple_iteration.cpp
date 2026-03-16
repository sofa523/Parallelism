// simple_iteration.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <vector>

using namespace std;

// Выделение памяти с проверкой
double* xmalloc(size_t size) {
    double* ptr = (double*)malloc(size);
    if (!ptr) {
        printf("Ошибка выделения памяти\n");
        exit(1);
    }
    return ptr;
}

// Решение СЛАУ методом простой итерации
int simple_iteration(double* A, double* b, double* x, int n, 
                     double epsilon, int max_iter) {
    double* x_new = xmalloc(sizeof(double) * n);
    int iter = 0;
    double error;
    
    do {
        error = 0.0;
        
        // Вычисление нового приближения
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sum += A[i * n + j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i * n + i];
        }
        
        // Вычисление погрешности
        error = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = fabs(x_new[i] - x[i]);
            if (diff > error) error = diff;
            x[i] = x_new[i];
        }
        
        iter++;
    } while (error > epsilon && iter < max_iter);
    
    free(x_new);
    return iter;
}

// Параллельная версия с OpenMP
int simple_iteration_omp(double* A, double* b, double* x, int n,
                         double epsilon, int max_iter) {
    double* x_new = xmalloc(sizeof(double) * n);
    int iter = 0;
    double error;
    
    do {
        error = 0.0;
        
        #pragma omp parallel
        {
            // Вычисление нового приближения
            #pragma omp for
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) {
                        sum += A[i * n + j] * x[j];
                    }
                }
                x_new[i] = (b[i] - sum) / A[i * n + i];
            }
            
            #pragma omp single
            {
                error = 0.0;
            }
            
            // Вычисление погрешности (максимальная разница)
            #pragma omp for reduction(max:error)
            for (int i = 0; i < n; i++) {
                double diff = fabs(x_new[i] - x[i]);
                if (diff > error) error = diff;
                x[i] = x_new[i];
            }
        }
        
        iter++;
    } while (error > epsilon && iter < max_iter);
    
    free(x_new);
    return iter;
}

int main(int argc, char **argv) {
    int n = 1000;  // Размерность системы
    double epsilon = 1e-6;
    int max_iter = 10000;
    
    if (argc > 1) n = atoi(argv[1]);
    
    printf("Simple iteration method for SLAE\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Memory used: %.2f MB\n", 
           ((double)(n * n + 2 * n) * sizeof(double)) / (1024.0 * 1024.0));
    
    // Выделение памяти
    double* A = xmalloc(sizeof(double) * n * n);
    double* b = xmalloc(sizeof(double) * n);
    double* x_serial = xmalloc(sizeof(double) * n);
    double* x_parallel = xmalloc(sizeof(double) * n);
    
    // Формирование матрицы A:
    // Элементы главной диагонали = 2.0, остальные = 1.0
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 2.0;
            } else {
                A[i * n + j] = 1.0;
            }
        }
    }
    
    // Формирование вектора b: все элементы равны N+1
    for (int i = 0; i < n; i++) {
        b[i] = n + 1;
    }
    
    // Начальное приближение x = 0
    for (int i = 0; i < n; i++) {
        x_serial[i] = 0.0;
        x_parallel[i] = 0.0;
    }
    
    printf("\n--- Решение системы ---\n");
    
    // Последовательный вариант
    double t = omp_get_wtime();
    int iter_serial = simple_iteration(A, b, x_serial, n, epsilon, max_iter);
    t = omp_get_wtime() - t;
    
    printf("Serial: iterations = %d, time = %.6f sec\n", iter_serial, t);
    
    // Проверка решения
    double error = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(x_serial[i] - 1.0);
        if (diff > error) error = diff;
    }
    printf("Max error of solution: %e\n", error);
    
    // Параллельный вариант для разного числа потоков
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_configs = sizeof(threads) / sizeof(threads[0]);
    
    printf("\n--- Масштабируемость ---\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Time (sec)", "Speedup");
    printf("----------------------------------------\n");
    
    for (int i = 0; i < num_configs; i++) {
        int num_threads = threads[i];
        omp_set_num_threads(num_threads);
        
        // Сброс начального приближения
        for (int j = 0; j < n; j++) {
            x_parallel[j] = 0.0;
        }
        
        double t_parallel = omp_get_wtime();
        int iter_parallel = simple_iteration_omp(A, b, x_parallel, n, epsilon, max_iter);
        t_parallel = omp_get_wtime() - t_parallel;
        
        double speedup = t / t_parallel;
        
        printf("%-10d %-15.6f %-15.2f\n", num_threads, t_parallel, speedup);
    }
    
    free(A);
    free(b);
    free(x_serial);
    free(x_parallel);
    
    return 0;
}