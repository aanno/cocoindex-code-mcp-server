#!/usr/bin/env python3
"""Test Java language support."""

from cocoindex_code_mcp_server.ast_visitor import analyze_code


class TestJavaSupport:
    """Test Java language analysis support."""

    def test_java_basic_analysis(self):
        """Test basic Java code analysis."""
        java_code = '''public class Calculator {
    private int value;
    
    public Calculator(int initialValue) {
        this.value = initialValue;
    }
    
    public int add(int x) {
        return value + x;
    }
    
    public static void main(String[] args) {
        Calculator calc = new Calculator(10);
        System.out.println(calc.add(5));
    }
}'''
        
        result = analyze_code(java_code, 'java', 'Calculator.java')
        
        assert result.get('success', False), f"Java analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"
        
        # Check for functions/methods
        functions = result.get('functions', [])
        classes = result.get('classes', [])
        
        # Should find at least some methods
        assert len(functions) > 0, f"Should find methods, got {functions}"
        
        # Check for expected methods
        expected_methods = {'add', 'main'}
        found_methods = set(functions)
        found_expected = found_methods & expected_methods
        
        assert len(found_expected) >= 1, f"Should find at least 1 expected method from {expected_methods}, got {functions}"
        
        # Check for class detection if supported
        if classes:
            assert 'Calculator' in classes, f"Should find Calculator class, got {classes}"

    def test_java_interface_support(self):
        """Test Java interface analysis."""
        java_code = '''public interface Drawable {
    void draw();
    default void print() {
        System.out.println("Drawing");
    }
}

public class Circle implements Drawable {
    @Override
    public void draw() {
        System.out.println("Drawing circle");
    }
}'''
        
        result = analyze_code(java_code, 'java', 'Drawable.java')
        
        assert result.get('success', False), "Java interface analysis should succeed"
        
        functions = result.get('functions', [])
        result.get('classes', [])
        
        # Should find methods
        assert len(functions) > 0, "Should find methods in interface and class"
        
        # Should find draw method
        assert 'draw' in functions, f"Should find draw method, got {functions}"

    def test_java_inheritance(self):
        """Test Java inheritance analysis."""
        java_code = '''public abstract class Animal {
    protected String name;
    
    public Animal(String name) {
        this.name = name;
    }
    
    public abstract void makeSound();
    
    public void sleep() {
        System.out.println(name + " is sleeping");
    }
}

public class Dog extends Animal {
    public Dog(String name) {
        super(name);
    }
    
    @Override
    public void makeSound() {
        System.out.println(name + " barks");
    }
    
    public void fetch() {
        System.out.println(name + " fetches");
    }
}'''
        
        result = analyze_code(java_code, 'java', 'Animal.java')
        
        assert result.get('success', False), "Java inheritance analysis should succeed"
        
        functions = result.get('functions', [])
        result.get('classes', [])
        
        # Should find methods
        assert len(functions) > 0, "Should find methods in inheritance hierarchy"
        
        # Should find some expected methods
        expected_methods = {'makeSound', 'sleep', 'fetch'}
        found_methods = set(functions)
        found_expected = found_methods & expected_methods
        
        assert len(found_expected) >= 1, f"Should find at least 1 method from {expected_methods}, got {functions}"

    def test_java_generics(self):
        """Test Java generics support."""
        java_code = '''import java.util.List;
import java.util.ArrayList;

public class GenericContainer<T> {
    private List<T> items;
    
    public GenericContainer() {
        this.items = new ArrayList<>();
    }
    
    public void add(T item) {
        items.add(item);
    }
    
    public T get(int index) {
        return items.get(index);
    }
    
    public static <U> void process(U item) {
        System.out.println(item.toString());
    }
}'''
        
        result = analyze_code(java_code, 'java', 'GenericContainer.java')
        
        assert result.get('success', False), "Java generics analysis should succeed"
        
        functions = result.get('functions', [])
        
        # Should find methods
        assert len(functions) > 0, "Should find methods in generic class"
        
        # Should find some basic methods
        expected_methods = {'add', 'get', 'process'}
        found_methods = set(functions)
        found_expected = found_methods & expected_methods
        
        assert len(found_expected) >= 1, f"Should find at least 1 method from {expected_methods}, got {functions}"