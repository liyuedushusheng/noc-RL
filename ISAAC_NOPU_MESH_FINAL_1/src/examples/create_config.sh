#!/bin/bash

# Specify the number of copies
i=4

# Source file
src_file="mesh88_lat_0"

# Check if source file exists
if [ ! -f "$src_file" ]; then
    echo "Source file $src_file does not exist!"
    exit 1
fi

# Copy files
for ((j=1; j<i; j++)); do
    cp "$src_file" "mesh88_lat_$j"
    echo "Created: mesh88_lat_$j"
done

