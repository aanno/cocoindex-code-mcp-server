/**
 * Test Java file for language analysis verification.
 * Tests function and class detection in Java code.
 */
package my.packages.structure;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

public class Main1 {

    public static class Person {
        private String name;
        private int age;

        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }

        public boolean isAdult() {
            return age >= 18;
        }

        public String greet() {
            return "Hello, I'm " + name + " and I'm " + age + " years old";
        }

        public String getName() { return name; }
        public int getAge() { return age; }
    }

    public static abstract class Shape {
        protected String color;

        public Shape(String color) {
            this.color = color;
        }

        public abstract double getArea();
        public abstract double getPerimeter();

        public String getColor() { return color; }
    }

    public static class Rectangle extends Shape {
        private double width;
        private double height;

        public Rectangle(String color, double width, double height) {
            super(color);
            this.width = width;
            this.height = height;
        }

        @Override
        public double getArea() {
            return width * height;
        }

        @Override
        public double getPerimeter() {
            return 2 * (width + height);
        }
    }

    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    public static int calculateSum(List<Integer> numbers) {
        int sum = 0;
        for (Integer num : numbers) {
            sum += num;
        }
        return sum;
    }

    public static <T> void printList(List<T> items) {
        for (T item : items) {
            System.out.println(item);
        }
    }

    public static boolean isPrime(int num) {
        if (num < 2) return false;
        for (int i = 2; i <= Math.sqrt(num); i++) {
            if (num % i == 0) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        Person person = new Person("Alice", 25);
        System.out.println(person.greet());
        System.out.println("Is adult: " + person.isAdult());

        System.out.println("Fibonacci 10: " + fibonacci(10));

        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        System.out.println("Sum: " + calculateSum(numbers));

        Rectangle rect = new Rectangle("red", 5.0, 3.0);
        System.out.println("Rectangle area: " + rect.getArea());
        System.out.println("Rectangle perimeter: " + rect.getPerimeter());
        System.out.println("Rectangle color: " + rect.getColor());

        System.out.println("Is 17 prime? " + isPrime(17));

        List<String> fruits = Arrays.asList("apple", "banana", "cherry");
        System.out.println("Fruits:");
        printList(fruits);
    }
}
