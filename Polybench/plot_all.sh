#!/bin/bash
timings=$1

python3 plot.py plot.pdf \
$timings/2mm.csv \
$timings/3mm.csv \
$timings/adi.csv \
$timings/atax.csv \
$timings/bicg.csv \
$timings/cholesky.csv \
$timings/correlation.csv \
$timings/covariance.csv \
$timings/deriche.csv \
$timings/doitgen.csv \
$timings/durbin.csv \
$timings/fdtd-2d.csv \
$timings/gemm.csv \
$timings/gramschmidt.csv \
$timings/gesummv.csv \
$timings/gemver.csv \
$timings/heat-3d.csv \
$timings/jacobi-1d.csv \
$timings/jacobi-2d.csv \
$timings/lu.csv \
$timings/ludcmp.csv \
$timings/mvt.csv \
$timings/seidel-2d.csv \
$timings/symm.csv \
$timings/syr2k.csv \
$timings/syrk.csv \
$timings/trisolv.csv \
$timings/trmm.csv
