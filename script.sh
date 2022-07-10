#!/bin/bash

# Usage: ./script.sh [cpp file]

# Exactly one global text symbol needed in the source file

# Settings
polygeist_bin=../../Polygeist/build/bin
out_dir=./out

# Create output directory
if [ ! -d $out_dir ]; then
  mkdir -p $out_dir;
fi

# Clear output directory
rm -rf $out_dir/*

# Get filename
src=$1
src_name=$(basename $1 .cpp)

# Find mangled name
clang++ -c $src -DUSE_MPI=0 -o $out_dir/$src_name.o
mangled_name=$(nm $out_dir/$src_name.o | grep -w T | cut -d ' ' -f3)
rm $out_dir/$src_name.o

# Polygeist cpp -> mlir
$polygeist_bin/cgeist -resource-dir=$(clang -print-resource-dir) \
  -function=$mangled_name -S -O3 -DUSE_MPI=0 $src &> $out_dir/$src_name.mlir

# Polygeist 
# $polygeist_bin/polygeist-opt --convert-polygeist-to-llvm 
