#!/bin/bash

rm -f ./output_file/ISAAC_baseline

#repeat 10 times
for i in {1..1000}
do
    sed -i "s/ifm_number = [0-9]*/ifm_number = $i/" ./examples/mesh88_lat
    echo "Running booksim - Attempt $i"
    ./booksim ./examples/mesh88_lat
    head -n 1 ./output_file/watch_file_196 >> ./output_file/ISAAC_baseline
    echo "--- End of Attempt $i ---" >> ./output_file/ISAAC_baseline
done
