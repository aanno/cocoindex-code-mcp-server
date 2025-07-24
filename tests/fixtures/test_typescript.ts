// Test TypeScript file for UniXcode embedding verification.
// This should use microsoft/unixcoder-base model.

interface User {
    id: number;
    name: string;
    email: string;
    isActive: boolean;
}

interface Repository<T> {
    findById(id: number): T | null;
    save(entity: T): void;
    findAll(): T[];
}

/**
 * Generic repository implementation
 */
class InMemoryRepository<T extends { id: number }> implements Repository<T> {
    private items: Map<number, T> = new Map();

    findById(id: number): T | null {
        return this.items.get(id) || null;
    }

    save(entity: T): void {
        this.items.set(entity.id, entity);
    }

    findAll(): T[] {
        return Array.from(this.items.values());
    }
}

/**
 * User service with type safety  
 */
class UserService {
    private repository: Repository<User>;

    constructor(repository: Repository<User>) {
        this.repository = repository;
    }

    createUser(name: string, email: string): User {
        const user: User = {
            id: Date.now(),
            name,
            email,
            isActive: true
        };
        
        this.repository.save(user);
        return user;
    }

    getActiveUsers(): User[] {
        return this.repository.findAll()
            .filter(user => user.isActive);
    }
}

// Usage example
const userRepo = new InMemoryRepository<User>();
const userService = new UserService(userRepo);

const newUser = userService.createUser("Alice Johnson", "alice@example.com");
console.log("Created user:", newUser);

const activeUsers = userService.getActiveUsers();
console.log("Active users:", activeUsers.length);