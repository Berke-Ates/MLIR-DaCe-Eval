#!/bin/bash

# Usage: ./script.sh [cpp file]

# Exactly one global text symbol needed in the source file
# Make sure the following are installed:
# clang & clang++
# Polygeist
# MLIR

# Settings
out_dir=./out

clang=$(which clang)                     || clang="NOT FOUND"
clangPP=$(which clang++)                 || clangPP="NOT FOUND"
mlir_opt=$(which mlir-opt)               || mlir_opt="NOT FOUND"
mlir_translate=$(which mlir-translate)   || mlir_translate="NOT FOUND"
cgeist=$(which cgeist)                   || cgeist="NOT FOUND"
polygeist_opt=$(which polygeist-opt)     || polygeist_opt="NOT FOUND"

echo "🔥 clang: $clang"
echo "   clang++: $clangPP"
echo "   mlir-opt: $mlir_opt"
echo "   mlir-translate: $mlir_translate"
echo "   cgeist: $cgeist"
echo "   polygeist-opt: $polygeist_opt"
echo "   output directory: $out_dir"
echo ""

set -e # Fail fast

# Check if tools exist
if [ "$clang" == "NOT FOUND" ] || \
   [ "$clangPP" == "NOT FOUND" ] || \
   [ "$mlir_opt" == "NOT FOUND" ] || \
   [ "$mlir_translate" == "NOT FOUND" ] || \
   [ "$cgeist" == "NOT FOUND" ] || \
   [ "$polygeist_opt" == "NOT FOUND" ]; then
    echo "Please make sure that the necessary tools are installed"
    exit 1;
fi

# Create output directory
if [ ! -d $out_dir ]; then
  mkdir -p $out_dir;
fi
rm -rf $out_dir/* # Clear output directory

# Get filename
src=$1
src_name=$(basename ${src%.*})
src_ext=${src##*.}
echo "🔥 Source: $src_name ($src)"

# Choose compiler based on C or C++
compiler=$clangPP
if [ "$src_ext" == "c" ]; then
  compiler=$clang;
fi
echo "   Using compiler: $(basename ${compiler%.*})"

# Find mangled name
$compiler -c $src -DUSE_MPI=0 -o $out_dir/$src_name.o
mangled_name=$(nm $out_dir/$src_name.o | grep -w T | cut -d ' ' -f3)
rm $out_dir/$src_name.o
echo "   Mangled function name: $mangled_name"
echo ""

# Polygeist c/cpp -> mlir
$cgeist -resource-dir=$(clang -print-resource-dir) \
  -function=$mangled_name -S -O3 -DUSE_MPI=0 $src > $out_dir/$src_name.mlir

# Lower Polygeist
if grep -q -i "polygeist" $out_dir/$src_name.mlir; then
  $polygeist_opt --convert-polygeist-to-llvm \
    $out_dir/$src_name.mlir > $out_dir/$src_name\_polygeist.mlir
  cp $out_dir/$src_name\_polygeist.mlir $out_dir/$src_name.mlir
  echo "🔥 Lowered Polygeist"
fi

# Lower affine 
if grep -q -i "affine" $out_dir/$src_name.mlir; then
  $mlir_opt --lower-affine $out_dir/$src_name.mlir > $out_dir/$src_name\_affine.mlir
  cp $out_dir/$src_name\_affine.mlir $out_dir/$src_name.mlir
  echo "🔥 Lowered Affine"
fi

# Lower to llvm 
$mlir_opt --lower-host-to-llvm $out_dir/$src_name.mlir > $out_dir/$src_name\_llvm.mlir
cp $out_dir/$src_name\_llvm.mlir $out_dir/$src_name.mlir
echo "🔥 Lowered to LLVM"

# Translate
$mlir_translate --mlir-to-llvmir $out_dir/$src_name.mlir > $out_dir/$src_name.ll
echo "🔥 Translated to LLVMIR"

# Compile
$clang -c -O3 $out_dir/$src_name.ll > $out_dir/$src_name.o
echo "🔥 Compiled"
