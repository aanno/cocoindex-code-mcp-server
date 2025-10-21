-- Haskell example with exactly 1 syntax error for testing error recovery
module TestModule where

-- Valid function
add :: Int -> Int -> Int
add x y = x + y

-- Error: missing closing parenthesis (should create 1-2 error nodes)
factorial :: Int -> Int
factorial n = n * factorial (n - 1

-- Valid function after error
multiply :: Int -> Int -> Int
multiply x y = x * y

-- Valid data type
data Color = Red | Green | Blue

-- Valid function
greeting :: String -> String
greeting name = "Hello, " ++ name ++ "!"
