#ifndef SERVER_H
#define SERVER_H

#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <cmath>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <vector>
#include <memory>

template<typename T>
class Server {
private:
    struct Task {
        size_t id;
        std::packaged_task<T()> task;
        
        Task(size_t i, std::packaged_task<T()>&& t) 
            : id(i), task(std::move(t)) {}   // перемещаем, не копируем
        
        Task(Task&& other) noexcept
            : id(other.id), task(std::move(other.task)) {}
        
        Task& operator=(Task&& other) noexcept {
            if (this != &other) {
                id = other.id;
                task = std::move(other.task);
            }
            return *this;
        }
        
        // запрещаем копирование
        Task(const Task&) = delete;
        Task& operator=(const Task&) = delete;
    };
    
    std::queue<Task> tasks;
    size_t next_id = 0;
    std::condition_variable cond_var;
    std::mutex mut;
    std::unordered_map<size_t, T> results;
    std::vector<std::jthread> workers;  // пул потоков
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> running{false};
    
    void process_tasks(std::stop_token stoken) {
        while (!stoken.stop_requested() && running) {
            std::unique_lock<std::mutex> lock(mut);
            
            cond_var.wait(lock, [this, &stoken] { 
                return !tasks.empty() || stoken.stop_requested() || !running; 
            });
            
            if (!running || stoken.stop_requested()) {
                break;
            }
            
            if (!tasks.empty()) {
                // Берем задачу из очереди
                Task task = std::move(tasks.front());
                tasks.pop();
                
                // Получаем future до выполнения
                std::future<T> future = task.task.get_future();
                lock.unlock();
                
                // Выполняем задачу
                task.task();
                
                // Получаем результат
                T result = future.get();
                
                lock.lock();
                results[task.id] = result;
            }
        }
    }
    
public:
    Server() = default;
    ~Server() { 
        stop(); 
    }
    
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    
    void start(int num_threads = std::thread::hardware_concurrency()) {
        if (running) {
            std::cout << "Server is already running\n";
            return;
        }
        
        if (num_threads == 0) num_threads = 2;
        if (num_threads > 16) num_threads = 16;
        
        running = true;
        stop_flag = false;
        workers.clear();
        
        for (int i = 0; i < num_threads; i++) {
            workers.emplace_back([this](std::stop_token stoken) { 
                process_tasks(stoken); 
            });
        }
        
        std::cout << "Thread Pool started with " << num_threads << " threads\n";
    }
    
    void stop() {
        if (!running) return;
        
        running = false;
        stop_flag = true;
        cond_var.notify_all();
        
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.request_stop();
            }
        }
        
        // Ждем завершения всех потоков
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        workers.clear();
        
        // Очищаем очередь задач
        std::lock_guard<std::mutex> lock(mut);
        while (!tasks.empty()) {
            tasks.pop();
        }
        
        std::cout << "Server stopped\n";
    }
    
    size_t add_task(std::function<T()> task) {
        std::packaged_task<T()> packaged_task(task);
        size_t id = next_id++;
        
        {
            std::lock_guard<std::mutex> lock(mut);
            tasks.emplace(id, std::move(packaged_task));
        }
        cond_var.notify_one();
        return id;
    }
    
    T request_result(size_t id_res) {
        std::unique_lock<std::mutex> lock(mut);
        
        // проверка наличия результата
        cond_var.wait(lock, [this, id_res] { 
            return results.find(id_res) != results.end(); 
        });
        
        T result = results[id_res];
        return result;
    }
    
    bool is_ready(size_t id_res) {
        std::lock_guard<std::mutex> lock(mut);
        return results.find(id_res) != results.end();
    }
    
    bool try_get_result(size_t id_res, T& out_result) {
        std::lock_guard<std::mutex> lock(mut);
        auto it = results.find(id_res);
        if (it != results.end()) {
            out_result = it->second;
            return true;
        }
        return false;
    }
};

#endif