-- Haskell example with 1-2 syntax errors for testing error recovery
module TestModule where

-- Valid function
add :: Int -> Int -> Int
add x y = x + y

-- Error 1: Invalid syntax in pattern matching (missing = after case)
factorial :: Int -> Int
factorial n = case n of
    0 -> 1
    _ -> n * factorial (n - 1  -- Error: missing closing parenthesis

-- Valid function
multiply :: Int -> Int -> Int
multiply x y = x * y

-- Valid data type
data Color = Red | Green | Blue

-- Error 2: Invalid where clause syntax
greeting :: String -> String
greeting name = "Hello, " ++ name ++ "!"
  where validName = if null name then "World" else name
        -- This line has correct syntax
