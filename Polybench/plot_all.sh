#!/bin/bash

benchmarks=$(find benchmarks/* -name '*.c' -not -path "benchmarks/utilities/*")

for filename in $benchmarks; do
    bname="$(basename $filename .c)"

    python3 plot.py timings_ault/$bname.csv plots/$bname.pdf $bname
done
