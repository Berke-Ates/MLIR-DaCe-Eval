#!/bin/bash

# Exactly one global text symbol needed in the source file

# Settings
src_name=mod_calcs # Name of the CPP file
polygeist_bin=../../Polygeist/build/bin

# Find mangled name
clang++ -c $src_name.cpp -DUSE_MPI=0
mangled_name=$(nm $src_name.o | grep -w T | cut -d ' ' -f3)
rm $src_name.o

# Polygeist cpp -> mlir
$polygeist_bin/cgeist -resource-dir=$(clang -print-resource-dir) \
  -function=$mangled_name -S -O3 -DUSE_MPI=0 $src_name.cpp &> $src_name.mlir

# Polygeist 
# $polygeist_bin/polygeist-opt --convert-polygeist-to-llvm 
