/**
 * Test C++ file for GraphCodeBERT embedding verification.
 * Tests function and class detection in C++ code.
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

class Person {
private:
    std::string name;
    int age;

public:
    Person(const std::string& name, int age) : name(name), age(age) {}

    bool isAdult() const {
        return age >= 18;
    }

    std::string greet() const {
        return "Hello, I'm " + name + " and I'm " + std::to_string(age) + " years old";
    }

    const std::string& getName() const { return name; }
    int getAge() const { return age; }
};

template<typename T>
class Calculator {
private:
    std::vector<std::string> history;

public:
    T add(T a, T b) {
        T result = a + b;
        history.push_back(std::to_string(a) + " + " + std::to_string(b) + " = " + std::to_string(result));
        return result;
    }

    T multiply(T a, T b) {
        T result = a * b;
        history.push_back(std::to_string(a) + " * " + std::to_string(b) + " = " + std::to_string(result));
        return result;
    }

    const std::vector<std::string>& getHistory() const {
        return history;
    }
};

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

template<typename T>
T calculateSum(const std::vector<T>& numbers) {
    T sum = T{};
    for (const auto& num : numbers) {
        sum += num;
    }
    return sum;
}

bool isPrime(int num) {
    if (num < 2) return false;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

int main() {
    Person person("Alice", 25);
    std::cout << person.greet() << std::endl;
    std::cout << "Is adult: " << (person.isAdult() ? "true" : "false") << std::endl;

    std::cout << "Fibonacci 10: " << fibonacci(10) << std::endl;

    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Sum: " << calculateSum(numbers) << std::endl;

    Calculator<int> calc;
    std::cout << "5 + 3 = " << calc.add(5, 3) << std::endl;
    std::cout << "4 * 7 = " << calc.multiply(4, 7) << std::endl;

    std::cout << "Is 17 prime? " << (isPrime(17) ? "true" : "false") << std::endl;

    // Lambda example
    auto square = [](int x) { return x * x; };
    std::cout << "Square of 5: " << square(5) << std::endl;

    return 0;
}
