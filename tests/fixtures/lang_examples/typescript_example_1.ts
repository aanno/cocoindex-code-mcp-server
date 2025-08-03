// Test TypeScript file for UniXcode embedding verification.
// This should use microsoft/unixcoder-base model.

interface User {
    id: number;
    name: string;
    email: string;
    age: number;
}

function fibonacci(n: number): number {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Person {
    private name: string;
    private age: number;

    constructor(name: string, age: number) {
        this.name = name;
        this.age = age;
    }

    greet(): string {
        return `Hello, I'm ${this.name} and I'm ${this.age} years old`;
    }

    isAdult(): boolean {
        return this.age >= 18;
    }

    getName(): string {
        return this.name;
    }

    getAge(): number {
        return this.age;
    }
}

function calculateSum(numbers: number[]): number {
    return numbers.reduce((sum, num) => sum + num, 0);
}

function processUsers(users: User[]): User[] {
    return users.filter(user => user.age >= 18);
}

function main(): void {
    const person = new Person("Alice", 25);
    console.log(person.greet());
    console.log("Is adult:", person.isAdult());

    console.log("Fibonacci 10:", fibonacci(10));

    const numbers = [1, 2, 3, 4, 5];
    console.log("Sum:", calculateSum(numbers));

    const users: User[] = [
        { id: 1, name: "Bob", email: "bob@example.com", age: 20 },
        { id: 2, name: "Charlie", email: "charlie@example.com", age: 16 }
    ];

    const adults = processUsers(users);
    console.log("Adult users:", adults.length);
}

// Call main if this is the entry point
if (require.main === module) {
    main();
}
