#!/bin/bash

# Usage: ./run_all.sh

benchmarks=("memory" "mem_line" "congrad_multi_field")

for i in ${!benchmarks[@]}; do
  ./run.sh ./benchmarks/${benchmarks[$i]}.c
  mv out/timings.txt timings/${benchmarks[$i]}.txt
  ./convert_csv.sh timings/${benchmarks[$i]}.txt 10
done
