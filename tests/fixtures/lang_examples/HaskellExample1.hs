-- Test Haskell file for fallback embedding verification.
-- This should use sentence-transformers/all-MiniLM-L6-v2 model.

module HaskellExample1 where

-- Simple data types
data Person = Person
    { personName :: String
    , personAge  :: Int
    } deriving (Show, Eq)

data Tree a = Leaf a | Branch (Tree a) (Tree a)
    deriving (Show, Eq)

-- Higher-order functions
fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

-- List processing with pattern matching
sumList :: [Int] -> Int
sumList []     = 0
sumList (x:xs) = x + sumList xs

-- Tree operations
treeMap :: (a -> b) -> Tree a -> Tree b
treeMap f (Leaf x)     = Leaf (f x)
treeMap f (Branch l r) = Branch (treeMap f l) (treeMap f r)

-- Function composition and currying
compose :: (b -> c) -> (a -> b) -> (a -> c)
compose f g x = f (g x)

addTen :: Int -> Int
addTen = (+) 10

multiplyByTwo :: Int -> Int
multiplyByTwo = (*) 2

-- Example usage
main :: IO ()
main = do
    let person = Person "Alice" 30
    putStrLn $ "Person: " ++ show person

    let numbers = [1, 2, 3, 4, 5]
    putStrLn $ "Sum of " ++ show numbers ++ " = " ++ show (sumList numbers)

    let tree = Branch (Leaf 1) (Branch (Leaf 2) (Leaf 3))
    let doubledTree = treeMap multiplyByTwo tree
    putStrLn $ "Original tree: " ++ show tree
    putStrLn $ "Doubled tree: " ++ show doubledTree

    let processNumber = compose addTen multiplyByTwo
    putStrLn $ "Process 5: " ++ show (processNumber 5)

    putStrLn $ "Fibonacci 10: " ++ show (fibonacci 10)
