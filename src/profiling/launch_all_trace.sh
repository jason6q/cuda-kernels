#!/bin/bash
# Basic script to just launch nsys on all test cases and store them here.

dir="${1:-.}"

echo "All files to nsys profile under $dir:"
echo "------------------------------"
find "$dir" -type f \( -name "test_*" ! -name "*.*" \) | while read -r f; do
    echo "$(basename "$f") — $(dirname "$f")" - $f
done

find "$dir" -type f \( -name "test_*" \) | while read -r f; do
    ./launch_trace.sh $f
done