#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <vector>

using namespace std;

int simple_iteration_omp(double* A, double* b, double* x, int n,
                         double epsilon, int max_iter) {
    double* x_new = (double*)malloc(sizeof(double) * n);
    int iter = 0;
     
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
            
    }
        
    iter++;
    
    free(x_new);
    return iter;
}

int main() {
    int n = 1000;  // Размерность системы
    double epsilon = 1e-6;
    int max_iter = 10000;
    double t = 99;
    
    // Выделение памяти
    double* A = (double*)malloc(sizeof(double) * n * n);
    double* b = (double*)malloc(sizeof(double) * n);
    double* x_parallel = (double*)malloc(sizeof(double) * n);
    
    // Формирование матрицы A: Элементы главной диагонали = 2.0, остальные = 1.0
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 2.0;
            } else {
                A[i * n + j] = 1.0;
            }
        }
    }
    
    
    for (int i = 0; i < n; i++)  b[i] = n + 1;       // Формирование вектора b: все элементы равны N+1
    for (int i = 0; i < n; i++) x_parallel[i] = 0.0;  // Начальное приближение x = 0
    

    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_configs = sizeof(threads) / sizeof(threads[0]);
    
    printf("\n--- Масштабируемость ---\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Time (sec)", "Speedup");
    printf("----------------------------------------\n");
    
    for (int i = 0; i < num_configs; i++) {
        int num_threads = threads[i];
        omp_set_num_threads(num_threads);
        
        // Сброс начального приближения
        for (int j = 0; j < n; j++) x_parallel[j] = 0.0;
        
        double t_parallel = omp_get_wtime();
        int iter_parallel = simple_iteration_omp(A, b, x_parallel, n, epsilon, max_iter);
        t_parallel = omp_get_wtime() - t_parallel;
        
        if(num_threads = 1) t = t_parallel;
        double speedup = t / t_parallel;
        
        printf("%-10d %-15.6f %-15.2f\n", num_threads, t_parallel, speedup);
    }
    
    free(A);
    free(b);
    free(x_parallel);
    
    return 0;
}