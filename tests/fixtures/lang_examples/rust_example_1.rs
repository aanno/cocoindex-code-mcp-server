// Test Rust file for UniXcode embedding verification.
// This should use microsoft/unixcoder-base model.

use std::collections::HashMap;

/// A simple struct to demonstrate Rust syntax
#[derive(Debug, Clone)]
pub struct Person {
    pub name: String,
    pub age: u32,
}

impl Person {
    /// Create a new Person instance
    pub fn new(name: String, age: u32) -> Self {
        Self { name, age }
    }

    /// Check if person is adult
    pub fn is_adult(&self) -> bool {
        self.age >= 18
    }
}

fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn main() {
    let person = Person::new("Alice".to_string(), 25);
    println!("Person: {:?}", person);
    println!("Is adult: {}", person.is_adult());

    let fib_10 = fibonacci(10);
    println!("Fibonacci(10): {}", fib_10);

    let mut scores: HashMap<String, i32> = HashMap::new();
    scores.insert("Alice".to_string(), 100);
    scores.insert("Bob".to_string(), 85);

    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }
}
