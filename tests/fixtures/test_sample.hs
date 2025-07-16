module Main where

import Data.List
import System.Environment

-- | Calculate factorial of a number
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- | Fibonacci sequence
fibonacci :: Integer -> Integer
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

-- | Data type for a simple binary tree
data Tree a = Empty | Node a (Tree a) (Tree a)
    deriving (Show, Eq)

-- | Insert element into binary search tree
insert :: Ord a => a -> Tree a -> Tree a
insert x Empty = Node x Empty Empty
insert x (Node a left right)
    | x <= a    = Node a (insert x left) right
    | otherwise = Node a left (insert x right)

-- | Search for element in binary search tree
search :: Ord a => a -> Tree a -> Bool
search _ Empty = False
search x (Node a left right)
    | x == a    = True
    | x < a     = search x left
    | otherwise = search x right

-- | Main function demonstrating various Haskell features
main :: IO ()
main = do
    args <- getArgs
    case args of
        []     -> putStrLn "Usage: program <number>"
        (n:_)  -> do
            let num = read n :: Integer
            putStrLn $ "Factorial of " ++ show num ++ " is " ++ show (factorial num)
            putStrLn $ "Fibonacci of " ++ show num ++ " is " ++ show (fibonacci num)
            
            -- Demonstrate tree operations
            let tree = foldr insert Empty [5, 3, 7, 2, 4, 6, 8]
            putStrLn $ "Tree: " ++ show tree
            putStrLn $ "Search for 4: " ++ show (search 4 tree)
            putStrLn $ "Search for 9: " ++ show (search 9 tree)