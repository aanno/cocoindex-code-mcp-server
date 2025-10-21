// Test JavaScript file for GraphCodeBERT embedding verification.
// This should use microsoft/graphcodebert-base model.

/**
 * Calculate factorial using recursion
 * @param {number} n - The number to calculate factorial for
 * @returns {number} The factorial result
 */
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

/**
 * A simple class to demonstrate JavaScript syntax
 */
class Calculator {
    constructor() {
        this.history = [];
    }

    /**
     * Add two numbers and record the operation
     * @param {number} a - First number
     * @param {number} b - Second number
     * @returns {number} Sum of a and b
     */
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }

    /**
     * Get calculation history
     * @returns {Array<string>} Array of calculation strings
     */
    getHistory() {
        return [...this.history];
    }
}

// Example usage
const calc = new Calculator();
console.log(`Factorial of 5: ${factorial(5)}`);
console.log(`Sum: ${calc.add(10, 20)}`);
console.log(`History: ${calc.getHistory()}`);

// Arrow function example
const isPrime = (num) => {
    if (num < 2) return false;
    for (let i = 2; i <= Math.sqrt(num); i++) {
        if (num % i === 0) return false;
    }
    return true;
};

console.log(`Is 17 prime? ${isPrime(17)}`);
