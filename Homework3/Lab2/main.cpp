#include "server.h"
#include <chrono>
#include <vector>
#include <memory>
#include <iostream>

void add_task_sin(Server<double>& server, int N, int client_id);
void add_task_sqrt(Server<double>& server, int N, int client_id);
void add_task_pow(Server<double>& server, int N, int client_id);

int main() {  
    const int N = 1000; // Количество задач для каждого клиента
    const int NUM_THREADS = std::thread::hardware_concurrency();
    
    std::cout << "System reports " << NUM_THREADS << " hardware threads\n";
    std::cout << "Using Thread Pool with " << NUM_THREADS << " worker threads\n\n";
    
    Server<double> server;
    server.start(NUM_THREADS);
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // клиенты
    std::thread client1(add_task_sin, std::ref(server), N, 1);
    std::thread client2(add_task_sqrt, std::ref(server), N, 2);
    std::thread client3(add_task_pow, std::ref(server), N, 3);
    
    client1.join();
    client2.join();
    client3.join();
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    server.stop();
    return 0;
}