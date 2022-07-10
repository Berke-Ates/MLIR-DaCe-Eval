#!/bin/bash

# Usage: ./script.sh [cpp file]

# Exactly one global text symbol needed in the source file

# Settings
out_dir=./out

clang=$(which clang)                     || clang="NOT FOUND"
clangPP=$(which clang++)                 || clangPP="NOT FOUND"
mlir_opt=$(which mlir-opt)               || mlir_opt="NOT FOUND"
mlir_translate=$(which mlir-translate)   || mlir_translate="NOT FOUND"
cgeist=$(which cgeist)                   || cgeist="NOT FOUND"
polygeist_opt=$(which polygeist-opt)     || polygeist_opt="NOT FOUND"
llc=$(which llc)                         || llc="NOT FOUND"

fmt_start="🔥 %-18s %s\n"
fmt_start_nl="\n🔥 %-18s %s\n"
fmt_list="   %-18s %s\n"
fmt_err="\n❌ %s\n"

printf "$fmt_start" "clang:" $clang
printf "$fmt_list" "clang++:" $clangPP
printf "$fmt_list" "mlir-opt:" $mlir_opt
printf "$fmt_list" "mlir-translate:" $mlir_translate
printf "$fmt_list" "cgeist:" $cgeist
printf "$fmt_list" "polygeist-opt:" $polygeist_opt
printf "$fmt_list" "llc:" $llc
printf "$fmt_list" "output directory:" $out_dir


set -e # Fail fast

# Check if tools exist
if [ "$clang" == "NOT FOUND" ] || \
   [ "$clangPP" == "NOT FOUND" ] || \
   [ "$mlir_opt" == "NOT FOUND" ] || \
   [ "$mlir_translate" == "NOT FOUND" ] || \
   [ "$cgeist" == "NOT FOUND" ] || \
   [ "$polygeist_opt" == "NOT FOUND" ]|| \
   [ "$llc" == "NOT FOUND" ]; then
    printf "$fmt_err" "Please make sure that the necessary tools are installed"
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
printf "$fmt_start_nl" "Source:" "$src_name ($src)"

# Choose compiler based on C or C++
compiler=$clangPP
if [ "$src_ext" == "c" ]; then
  compiler=$clang;
fi
printf "$fmt_list" "Using compiler:" "$(basename ${compiler%.*})"

# Find mangled name
$compiler -c $src -DUSE_MPI=0 -o $out_dir/$src_name.o
mangled_name=$(nm $out_dir/$src_name.o | grep -w T | cut -d ' ' -f3)
rm $out_dir/$src_name.o
printf "$fmt_list\n" "Function name:" "$mangled_name"

# Polygeist c/cpp -> mlir
$cgeist -resource-dir=$(clang -print-resource-dir) \
  -function=$mangled_name -S -O3 -DUSE_MPI=0 $src > $out_dir/$src_name.mlir
cp $out_dir/$src_name.mlir $out_dir/$src_name\_cgeist.mlir

# Lower affine 
if grep -q -i "affine" $out_dir/$src_name.mlir; then
  $mlir_opt --lower-affine $out_dir/$src_name.mlir > $out_dir/$src_name\_affine.mlir
  cp $out_dir/$src_name\_affine.mlir $out_dir/$src_name.mlir
  printf "$fmt_start" "Lowered Affine"
fi

# Lower Polygeist
if grep -q -i "polygeist" $out_dir/$src_name.mlir; then
  $polygeist_opt --convert-polygeist-to-llvm \
    $out_dir/$src_name.mlir > $out_dir/$src_name\_polygeist.mlir
  cp $out_dir/$src_name\_polygeist.mlir $out_dir/$src_name.mlir
  printf "$fmt_start" "Lowered Polygeist"
fi

# Lower to llvm 
$mlir_opt --lower-host-to-llvm $out_dir/$src_name.mlir > $out_dir/$src_name\_llvm.mlir
cp $out_dir/$src_name\_llvm.mlir $out_dir/$src_name.mlir
printf "$fmt_start" "Lowered to LLVM"

# Translate
$mlir_translate --mlir-to-llvmir $out_dir/$src_name.mlir > $out_dir/$src_name.ll
printf "$fmt_start" "Translated to LLVMIR"

# Compile
$llc -O3 --relocation-model=pic $out_dir/$src_name.ll -o $out_dir/$src_name.s
printf "$fmt_start" "Compiled"

# Assemble
$clang -c -O3 $out_dir/$src_name.s -o $out_dir/$src_name.o
printf "$fmt_start" "Assembled"
