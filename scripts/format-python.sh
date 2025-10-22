#!/bin/bash -x

# Check for tool availability
for tool in isort autoflake8 autopep8; do
    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool not found. Please install it before running this script."
        exit 1
    fi
done

# If no arguments provided, print usage
if [ $# -eq 0 ]; then
    echo "Usage: $0 folder1 [folder2 ...]"
    exit 1
fi

# Loop over all directories/files passed as arguments
for target in "$@"; do
    echo "Processing: $target"

    # Run isort recursively on all Python files in target
    isort "$target"
    # --profile black

    ruff check --fix "$target"
    ruff format "$target"

    pydocstyle "$target"

    # Run autoflake8 recursively - remove unused imports and variables, in-place
    autoflake8 --in-place --recursive "$target"
    # --remove-unused-variables --remove-all-unused-imports

    # Run autopep8 recursively - in-place with aggressive fixes
    autopep8 --in-place --recursive --aggressive "$target"
done

echo "Done formatting."
