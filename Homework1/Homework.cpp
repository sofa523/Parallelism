#include <iostream>
#include <cmath>
#include <vector>

#if ARRAY_TYPE == 1  // float
    using ArrayType = float;
    const char* typeName = "float";
#else
    using ArrayType = double;
    const char* typeName = "double";
#endif

int main() {
    const int n = 10000000; // 10^7 элементов
    // const ArrayType two_pi = static_cast<ArrayType>(2.0 * M_PI);
    const ArrayType two_pi = static_cast<ArrayType>(6.283185307179586);
    
    std::cout << "Type array: " << typeName << std::endl;
    
    // создаем и заполняем массив значениями синуса
    std::vector<ArrayType> array(n);
    for (int i = 0; i < n; ++i) {
        ArrayType x = static_cast<ArrayType>(i) / static_cast<ArrayType>(n) * 2.0 * two_pi;
        array[i] = std::sin(x);
    }
    
    // сумма
    ArrayType sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += array[i];
    }
    std::cout << "Sum = " << sum << std::endl;
    
    return 0;
}