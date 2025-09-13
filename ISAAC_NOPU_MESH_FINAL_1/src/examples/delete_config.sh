#!/bin/bash

# Specify the range of files to delete
i=1
j=200

# Delete files
for ((k=i; k<=j; k++)); do
    file="mesh88_lat_$k"
    if [ -f "$file" ]; then
        rm "$file"
        echo "Deleted: $file"
    else
        echo "File not found: $file"
    fi
done

