#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <cstring>


double func(double x) {
    return exp(-x * x);  // Гауссовское распределение
}

double integrate_parallel(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        
        double local_sum = 0.0;
        
        for (int i = lb; i <= ub; i++) {
            local_sum += f(a + h * (i + 0.5));
        }
        
        #pragma omp atomic
        sum += local_sum;
    }
    
    return sum * h;
}

int main() {
    double a = -4.0, b = 4.0;
    int nsteps = 40000000;
    double serial_time = -100;
    
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_configs = sizeof(threads) / sizeof(threads[0]);
    
    printf("\n--- Масштабируемость (nsteps = %d) ---\n", nsteps);
    printf("%-10s %-15s %-15s\n", "Threads", "Time (sec)", "Speedup");
    printf("----------------------------------------\n");
    
    for (int i = 0; i < num_configs; i++) {
        int num_threads = threads[i];
        omp_set_num_threads(num_threads);
        
        double t_parallel = omp_get_wtime();
        double result_parallel = integrate_parallel(func, a, b, nsteps);
        t_parallel = omp_get_wtime() - t_parallel;

        if (num_threads == 1) serial_time = t_parallel;
        double speedup = serial_time / t_parallel;
        
        printf("%-10d %-15.6f %-15.2f\n", 
               num_threads, t_parallel, speedup);
    }
    
    return 0;
}