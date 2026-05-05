#include "server.h"
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <thread>

// генератор случайных чисел
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dis(0.1, 1000.0);

// Функции задач
double func_sin(double x) { 
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    return std::sin(x); 
}

double func_sqrt(double x) { 
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    return std::sqrt(x); 
}

double func_pow(double x, double y) { 
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    return std::pow(x, y); 
}

void add_task_sin(Server<double>& server, int N, int client_id) {
    // Создаем директорию results если её нет
    system("mkdir -p results");
    
    std::string filename = "results/client_" + std::to_string(client_id) + "_sin.txt";
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    std::vector<size_t> ids;
    std::vector<double> args;
    ids.reserve(N);
    args.reserve(N);
    
    file << "=== Client " << client_id << " - SIN Tasks ===\n";
    file << std::fixed << std::setprecision(10);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Добавляем задачи
    for (int i = 0; i < N; i++) {
        double arg = dis(gen);
        size_t task_id = server.add_task([arg]() { return func_sin(arg); });
        ids.push_back(task_id);
        args.push_back(arg);
    }
    
    // Собираем результаты
    for (size_t i = 0; i < ids.size(); i++) {
        double res = server.request_result(ids[i]);
        file << "Task " << std::setw(6) << ids[i] 
             << ": sin(" << std::setw(12) << args[i] 
             << ") = " << std::setw(15) << res << "\n";
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    file << "\n=== Summary ===\n";
    file << "Total tasks: " << N << "\n";
    file << "Total time: " << duration.count() << " ms\n";
    if (N > 0) {
        file << "Average time per task: " << (duration.count() / N) << " ms\n";
    }
    file.close();
    
    std::cout << "Client " << client_id << " (sin) completed " << N << " tasks in " 
              << duration.count() << " ms\n";
}

void add_task_sqrt(Server<double>& server, int N, int client_id) {
    system("mkdir -p results");
    
    std::string filename = "results/client_" + std::to_string(client_id) + "_sqrt.txt";
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    std::vector<size_t> ids;
    std::vector<double> args;
    ids.reserve(N);
    args.reserve(N);
    
    file << "=== Client " << client_id << " - SQRT Tasks ===\n";
    file << std::fixed << std::setprecision(10);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < N; i++) {
        double arg = dis(gen);
        size_t task_id = server.add_task([arg]() { return func_sqrt(arg); });
        ids.push_back(task_id);
        args.push_back(arg);
    }
    
    for (size_t i = 0; i < ids.size(); i++) {
        double res = server.request_result(ids[i]);
        file << "Task " << std::setw(6) << ids[i] 
             << ": sqrt(" << std::setw(12) << args[i] 
             << ") = " << std::setw(15) << res << "\n";
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    file << "\n=== Summary ===\n";
    file << "Total tasks: " << N << "\n";
    file << "Total time: " << duration.count() << " ms\n";
    if (N > 0) {
        file << "Average time per task: " << (duration.count() / N) << " ms\n";
    }
    file.close();
    
    std::cout << "Client " << client_id << " (sqrt) completed " << N << " tasks in " 
              << duration.count() << " ms\n";
}

void add_task_pow(Server<double>& server, int N, int client_id) {
    // Создаем директорию results если её нет
    system("mkdir -p results");
    
    std::string filename = "results/client_" + std::to_string(client_id) + "_pow.txt";
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    std::vector<size_t> ids;
    std::vector<std::pair<double, double>> args;
    ids.reserve(N);
    args.reserve(N);
    
    file << "=== Client " << client_id << " - POW Tasks ===\n";
    file << std::fixed << std::setprecision(10);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < N; i++) {
        double base = dis(gen);
        double exp = dis(gen) / 10.0;
        size_t task_id = server.add_task([base, exp]() { return func_pow(base, exp); });
        ids.push_back(task_id);
        args.push_back({base, exp});
    }
    
    for (size_t i = 0; i < ids.size(); i++) {
        double res = server.request_result(ids[i]);
        file << "Task " << std::setw(6) << ids[i] 
             << ": pow(" << std::setw(10) << args[i].first 
             << ", " << std::setw(10) << args[i].second 
             << ") = " << std::setw(15) << res << "\n";
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    file << "\n=== Summary ===\n";
    file << "Total tasks: " << N << "\n";
    file << "Total time: " << duration.count() << " ms\n";
    if (N > 0) {
        file << "Average time per task: " << (duration.count() / N) << " ms\n";
    }
    file.close();
    
    std::cout << "Client " << client_id << " (pow) completed " << N << " tasks in " 
              << duration.count() << " ms\n";
}