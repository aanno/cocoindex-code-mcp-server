package my.packages.structure;

/**
 * Test Kotlin file for language analysis verification.
 * Tests function and class detection in Kotlin code.
 */

data class Person(val name: String, val age: Int) {
    fun isAdult(): Boolean = age >= 18

    fun greet(): String = "Hello, I'm $name and I'm $age years old"
}

sealed class Result<out T> {
    data class Success<out T>(val value: T) : Result<T>()
    data class Error(val message: String) : Result<Nothing>()
}

fun fibonacci(n: Int): Int {
    return when (n) {
        0 -> 0
        1 -> 1
        else -> fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fun processResult(result: Result<Int>): String {
    return when (result) {
        is Result.Success -> "Value: ${result.value}"
        is Result.Error -> "Error: ${result.message}"
    }
}

fun calculateSum(numbers: List<Int>): Int {
    return numbers.fold(0) { acc, num -> acc + num }
}

class Calculator {
    private var history: MutableList<String> = mutableListOf()

    fun add(a: Int, b: Int): Int {
        val result = a + b
        history.add("$a + $b = $result")
        return result
    }

    fun multiply(a: Int, b: Int): Int {
        val result = a * b
        history.add("$a * $b = $result")
        return result
    }

    fun getHistory(): List<String> = history.toList()
}

fun main() {
    val person = Person("Alice", 25)
    println(person.greet())
    println("Is adult: ${person.isAdult()}")

    println("Fibonacci 10: ${fibonacci(10)}")

    val numbers = listOf(1, 2, 3, 4, 5)
    println("Sum: ${calculateSum(numbers)}")

    val calc = Calculator()
    println("5 + 3 = ${calc.add(5, 3)}")
    println("4 * 7 = ${calc.multiply(4, 7)}")

    val successResult = Result.Success(42)
    val errorResult = Result.Error("Something went wrong")
    println(processResult(successResult))
    println(processResult(errorResult))
}
