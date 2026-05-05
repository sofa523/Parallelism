#include "server.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <cmath>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

struct TestResult {
    std::string client_name;
    int total_tasks = 0;
    int correct_tasks = 0;
    int wrong_tasks = 0;
    double max_error = 0.0;
    double avg_error = 0.0;
};

bool test_sin_file(const std::string& filename, TestResult& result) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    std::regex sin_regex(R"(Task\s+(\d+):\s+sin\(([\d.eE+-]+)\)\s+=\s+([\d.eE+-]+))");
    std::smatch match;
    std::string line;
    
    double total_error = 0.0;
    result.client_name = "SIN";
    result.total_tasks = 0;
    result.correct_tasks = 0;
    result.wrong_tasks = 0;
    result.max_error = 0.0;
    
    const double epsilon = 1e-8;
    
    while (std::getline(file, line)) {
        if (std::regex_search(line, match, sin_regex)) {
            result.total_tasks++;
            double arg = std::stod(match[2]);
            double computed = std::stod(match[3]);
            double expected = std::sin(arg);
            double error = std::abs(computed - expected);
            
            total_error += error;
            if (error > result.max_error) result.max_error = error;
            
            if (error < epsilon) {
                result.correct_tasks++;
            } else {
                result.wrong_tasks++;
                std::cout << "  Error in sin(" << arg << "): computed=" << computed 
                         << ", expected=" << expected << ", diff=" << error << std::endl;
            }
        }
    }
    
    if (result.total_tasks > 0) {
        result.avg_error = total_error / result.total_tasks;
    }
    
    return result.wrong_tasks == 0;
}

bool test_sqrt_file(const std::string& filename, TestResult& result) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    std::regex sqrt_regex(R"(Task\s+(\d+):\s+sqrt\(([\d.eE+-]+)\)\s+=\s+([\d.eE+-]+))");
    std::smatch match;
    std::string line;
    
    double total_error = 0.0;
    result.client_name = "SQRT";
    result.total_tasks = 0;
    result.correct_tasks = 0;
    result.wrong_tasks = 0;
    result.max_error = 0.0;
    
    const double epsilon = 1e-8;
    
    while (std::getline(file, line)) {
        if (std::regex_search(line, match, sqrt_regex)) {
            result.total_tasks++;
            double arg = std::stod(match[2]);
            double computed = std::stod(match[3]);
            double expected = std::sqrt(arg);
            double error = std::abs(computed - expected);
            
            total_error += error;
            if (error > result.max_error) result.max_error = error;
            
            if (error < epsilon) {
                result.correct_tasks++;
            } else {
                result.wrong_tasks++;
                std::cout << "  Error in sqrt(" << arg << "): computed=" << computed 
                         << ", expected=" << expected << ", diff=" << error << std::endl;
            }
        }
    }
    
    if (result.total_tasks > 0) {
        result.avg_error = total_error / result.total_tasks;
    }
    
    return result.wrong_tasks == 0;
}

bool test_pow_file(const std::string& filename, TestResult& result) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    std::regex pow_regex(R"(Task\s+(\d+):\s+pow\(([\d.eE+-]+),\s+([\d.eE+-]+)\)\s+=\s+([\d.eE+-]+))");
    std::smatch match;
    std::string line;
    
    result.client_name = "POW";
    result.total_tasks = 0;
    result.correct_tasks = 0;
    result.wrong_tasks = 0;
    result.max_error = 0.0;
    
    const double eps_small = 1e-8; 
    const double eps_large = 1e-5; 
    
    int displayed_errors = 0;
    const int MAX_DISPLAY = 20;
    
    while (std::getline(file, line)) {
        if (std::regex_search(line, match, pow_regex)) {
            result.total_tasks++;
            double base = std::stod(match[2]);
            double exp = std::stod(match[3]);
            double computed = std::stod(match[4]);
            double expected = std::pow(base, exp);
            
            double abs_error = std::abs(computed - expected);
            double rel_error = abs_error / (std::abs(expected) + 1e-300);
            
            if (rel_error > result.max_error) result.max_error = rel_error;
            
            // Адаптивный порог в зависимости от величины результата
            double threshold;
            double expected_abs = std::abs(expected);
            
            if (expected_abs < 1e-50) {
                // Очень маленькие числа - используем абсолютную погрешность
                threshold = (abs_error < 1e-40) ? 0 : 1;
            } else if (expected_abs < 1e10) {
                // Нормальные числа
                threshold = (rel_error < 1e-10) ? 0 : 1;
            } else if (expected_abs < 1e50) {
                // Большие числа
                threshold = (rel_error < 1e-9) ? 0 : 1;
            } else if (expected_abs < 1e100) {
                // Очень большие числа
                threshold = (rel_error < 1e-8) ? 0 : 1;
            } else if (expected_abs < 1e200) {
                // Экстремально большие числа
                threshold = (rel_error < 1e-7) ? 0 : 1;
            } else {
                // Гигантские числа (e+200 и больше)
                threshold = (rel_error < 1e-6) ? 0 : 1;
            }
            
            if (threshold == 0) {
                result.correct_tasks++;
            } else {
                result.wrong_tasks++;
                if (displayed_errors < MAX_DISPLAY) {
                    std::cout << "  Error in pow(" << base << ", " << exp 
                             << "): computed=" << computed 
                             << ", expected=" << expected 
                             << ", rel_error=" << rel_error
                             << " (magnitude: " << std::log10(expected_abs) << ")" << std::endl;
                    displayed_errors++;
                }
            }
        }
    }
    
    if (result.total_tasks > 0) {
        result.avg_error = result.max_error / result.total_tasks;
    }
    
    std::cout << "    Relative error threshold: adaptive (1e-6 to 1e-10)\n";
    std::cout << "    Wrong tasks: " << result.wrong_tasks;
    if (result.wrong_tasks > MAX_DISPLAY) {
        std::cout << " (" << (result.wrong_tasks - MAX_DISPLAY) << " more not shown)";
    }
    std::cout << std::endl;
    
    double pass_rate = (result.correct_tasks * 100.0) / result.total_tasks;
    return pass_rate >= 99;
}

int main() {
    std::cout << "Testing Client-Server Application\n";
    
    fs::create_directories("results");
    
    std::vector<std::pair<std::string, std::function<bool(const std::string&, TestResult&)>>> tests = {
        {"results/client_1_sin.txt", test_sin_file},
        {"results/client_2_sqrt.txt", test_sqrt_file},
        {"results/client_3_pow.txt", test_pow_file}
    };
    
    int passed = 0;
    int total = 0;
    std::vector<TestResult> results;
    
    for (const auto& [filename, test_func] : tests) {
        std::cout << "Testing " << filename << "...\n";
        
        if (fs::exists(filename)) {
            TestResult result;
            if (test_func(filename, result)) {
                std::cout << "PASSED\n";
                passed++;
            } else {
                std::cout << "FAILED\n";
            }
            total++;
            results.push_back(result);
            
            std::cout << "    Total tasks: " << result.total_tasks << "\n";
            std::cout << "    Correct: " << result.correct_tasks << "\n";
            std::cout << "    Wrong: " << result.wrong_tasks << "\n";
        } else {
            total++;
        }
    }
    
    std::cout << "Test Summary: " << passed << "/" << total << " passed\n";
    
    return (passed == total) ? 0 : 1;
}