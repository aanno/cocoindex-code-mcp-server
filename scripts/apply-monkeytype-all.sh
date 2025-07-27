#!/bin/bash

# List all modules with collected traces
modules=$(monkeytype list-modules)

if [ -z "$modules" ]; then
    echo "No modules found in MonkeyType database."
    exit 0
fi

echo "Applying MonkeyType annotations to modules:"
echo "$modules"
echo

# Apply annotations to each module
for mod in $modules; do
    echo "Applying annotations to module: $mod"
    monkeytype apply "$mod"
done

echo "All modules processed."
