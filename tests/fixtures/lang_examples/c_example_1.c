/* Test C file for baseline verification */

#include <stdio.h>
#include <stdlib.h>

// Simple data structures
typedef struct {
    int x;
    int y;
} Point;

typedef enum {
    RED,
    GREEN,
    BLUE
} Color;

// Function declarations
int add(int a, int b);
void print_point(Point p);
Point create_point(int x, int y);
Color get_default_color(void);

// Function implementations
int add(int a, int b) {
    return a + b;
}

void print_point(Point p) {
    printf("Point(%d, %d)\n", p.x, p.y);
}

Point create_point(int x, int y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

Color get_default_color(void) {
    return RED;
}

int main() {
    Point p = create_point(10, 20);
    print_point(p);

    int sum = add(5, 3);
    printf("Sum: %d\n", sum);

    Color c = get_default_color();
    printf("Default color: %d\n", c);

    return 0;
}
