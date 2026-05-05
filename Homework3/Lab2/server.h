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
        std::function<T()> func;
        std::promise<T> promise;
        
        Task(size_t i, std::function<T()> f) 
            : id(i), func(std::move(f)) {}
        
        Task(Task&& other) noexcept
            : id(other.id), 
              func(std::move(other.func)),
              promise(std::move(other.promise)) {}
        
        Task& operator=(Task&& other) noexcept {
            if (this != &other) {
                id = other.id;
                func = std::move(other.func);
                promise = std::move(other.promise);
            }
            return *this;
        }
        
        // запрещаем копирование
        Task(const Task&) = delete;
        Task& operator=(const Task&) = delete;
    };
    
    std::queue<Task> tasks;
    std::atomic<size_t> next_id{0};
    std::condition_variable cond_var;
    std::mutex mut;
    std::unordered_map<size_t, std::future<T>> futures;
    std::vector<std::thread> workers;  // используем std::thread вместо std::jthread для совместимости
    std::atomic<bool> running{false};
    std::atomic<bool> stop_flag{false};
    
    void process_tasks() {
        while (running || !stop_flag) {
            std::unique_lock<std::mutex> lock(mut);
            
            cond_var.wait(lock, [this] { 
                return !tasks.empty() || !running || stop_flag; 
            });
            
            if ((!running || stop_flag) && tasks.empty()) {
                break;
            }
            
            if (!tasks.empty()) {
                // Берем задачу из очереди
                Task task = std::move(tasks.front());
                tasks.pop();
                
                // Сохраняем promise и future
                std::promise<T> promise = std::move(task.promise);
                
                lock.unlock();
                
                try {
                    // Выполняем задачу
                    T result = task.func();
                    // Устанавливаем результат
                    promise.set_value(result);
                } catch (...) {
                    // В случае ошибки устанавливаем исключение
                    promise.set_exception(std::current_exception());
                }
                
                lock.lock();
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
            workers.emplace_back([this]() { 
                process_tasks(); 
            });
        }
        
        std::cout << "Thread Pool started with " << num_threads << " threads\n";
    }
    
    void stop() {
        if (!running) return;
        
        running = false;
        stop_flag = true;
        cond_var.notify_all();
        
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
        
        // Очищаем futures
        futures.clear();
        
        std::cout << "Server stopped\n";
    }
    
    size_t add_task(std::function<T()> task) {
        size_t id = next_id++;
        
        std::promise<T> promise;
        std::future<T> future = promise.get_future();
        
        {
            std::lock_guard<std::mutex> lock(mut);
            tasks.emplace(id, std::move(task));
            tasks.back().promise = std::move(promise);
            futures[id] = std::move(future);
        }
        
        cond_var.notify_one();
        return id;
    }
    
    T request_result(size_t id_res) {
        std::unique_lock<std::mutex> lock(mut);
        
        auto it = futures.find(id_res);
        if (it == futures.end()) {
            lock.unlock();
            throw std::runtime_error("No such task id: " + std::to_string(id_res));
        }
        
        std::future<T> fut = std::move(it->second);
        futures.erase(it);
        
        lock.unlock();
        
        if (fut.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
            throw std::runtime_error("Timeout waiting for result " + std::to_string(id_res));
        }
        
        return fut.get();
    }
    
    bool is_ready(size_t id_res) {
        std::lock_guard<std::mutex> lock(mut);
        auto it = futures.find(id_res);
        if (it == futures.end()) {
            return false;
        }
        
        // Проверяем, готов ли результат без блокировки
        return it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    
    bool try_get_result(size_t id_res, T& out_result) {
        std::unique_lock<std::mutex> lock(mut);
        auto it = futures.find(id_res);
        if (it == futures.end()) {
            return false;
        }
        
        // Проверяем, готов ли результат
        if (it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            std::future<T> fut = std::move(it->second);
            futures.erase(it);
            lock.unlock();
            
            out_result = fut.get();
            return true;
        }
        
        return false;
    }
    
    size_t pending_tasks_count() {
        std::lock_guard<std::mutex> lock(mut);
        return tasks.size();
    }
    
    size_t pending_results_count() {
        std::lock_guard<std::mutex> lock(mut);
        return futures.size();
    }
};

#endif // SERVER_H