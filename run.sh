#!/bin/bash

# Usage: ./script.sh [cpp file]

# Exactly one global text symbol needed in the source file

# Settings
driver=lulesh_driver.cpp
opt_lvl=O0
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
printf "$fmt_list" "output dir:" $out_dir
printf "$fmt_list" "opt lvl:" $opt_lvl


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
src_dir=$(dirname $src)
printf "$fmt_start_nl" "Source:" "$src_name ($src)"

# Choose compiler based on C or C++
compiler=$clangPP
if [ "$src_ext" == "c" ]; then
  compiler=$clang;
fi
printf "$fmt_list" "Using compiler:" "$(basename ${compiler%.*})"

# Replace inserts
if grep -q -i "#insert" $src; then
  cp $src $out_dir/$src_name\_inserted.$src_ext
  inserts=$(grep -E "^#insert"  $src)

  while IFS= read -r line; do
    fname=$(echo $line | cut -d ' ' -f 2)

    if [ ! -f $src_dir/$fname ]; then
      printf "$fmt_err" "Insert file not found: $src_dir/$fname"
      exit 1
    fi

    sed -i -e "/$line/r$src_dir/$fname" $out_dir/$src_name\_inserted.$src_ext
    sed -i -e "s/$line//" $out_dir/$src_name\_inserted.$src_ext

    printf "$fmt_list" "Replaced:" "$line"
  done <<< "$inserts"

  src=$out_dir/$src_name\_inserted.$src_ext
fi

# Find mangled name
$compiler -c $src -DUSE_MPI=0 -o $out_dir/$src_name.o
mangled_name=$(nm $out_dir/$src_name.o | grep $src_name | cut -d ' ' -f3)
rm $out_dir/$src_name.o
printf "$fmt_list\n" "Function name:" "$mangled_name"

# Polygeist c/cpp -> mlir
$cgeist -resource-dir=$(clang -print-resource-dir) -function=$mangled_name \
  -S --memref-fullrank --memref-abi=0 --struct-abi=0 -$opt_lvl -DUSE_MPI=0 \
  --raise-scf-to-affine $src > $out_dir/$src_name.mlir
cp $out_dir/$src_name.mlir $out_dir/$src_name\_cgeist.mlir
printf "$fmt_start" "Lowered to:" "MLIR"

# Lower affine 
if grep -q -i "affine" $out_dir/$src_name.mlir; then
  $mlir_opt --lower-affine --allow-unregistered-dialect \
    $out_dir/$src_name.mlir > $out_dir/$src_name\_affine.mlir
  cp $out_dir/$src_name\_affine.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Lowered:" "Affine"
fi

# Lower scf 
if grep -q -i "scf" $out_dir/$src_name.mlir; then
  $mlir_opt --convert-scf-to-cf --allow-unregistered-dialect \
     $out_dir/$src_name.mlir > $out_dir/$src_name\_scf.mlir
  cp $out_dir/$src_name\_scf.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Lowered:" "SCF"
fi

# Lower cf
if grep -q -i "cf" $out_dir/$src_name.mlir; then
  $mlir_opt --convert-cf-to-llvm --allow-unregistered-dialect \
     $out_dir/$src_name.mlir > $out_dir/$src_name\_cf.mlir
  cp $out_dir/$src_name\_cf.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Lowered:" "CF"
fi

# Lower Polygeist
if grep -q -i "polygeist" $out_dir/$src_name.mlir; then
  $polygeist_opt --convert-polygeist-to-llvm \
    $out_dir/$src_name.mlir > $out_dir/$src_name\_polygeist.mlir
  cp $out_dir/$src_name\_polygeist.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Lowered:" "Polygeist"
fi

# Lower to llvm 
$mlir_opt --lower-host-to-llvm $out_dir/$src_name.mlir > $out_dir/$src_name\_llvm.mlir
cp $out_dir/$src_name\_llvm.mlir $out_dir/$src_name.mlir
printf "$fmt_list" "Lowered to:" "LLVM"

# Rename interface
# Unsure if needed
# if grep -q -i "_mlir_ciface_" $out_dir/$src_name.mlir; then
#   sed -i -e "s/_mlir_ciface_$mangled_name/_mlir_ciface_/g" $out_dir/$src_name.mlir
#   sed -i -e "s/$mangled_name/$mangled_name\_renamed/g" $out_dir/$src_name.mlir
#   sed -i -e "s/_mlir_ciface_/$mangled_name/g" $out_dir/$src_name.mlir
#   printf "$fmt_list" "Renamed Interface"
# fi

# Translate
$mlir_translate --mlir-to-llvmir $out_dir/$src_name.mlir > $out_dir/$src_name.ll
printf "$fmt_list" "Translated to:" "LLVMIR"

# Compile
$llc -$opt_lvl --relocation-model=pic $out_dir/$src_name.ll -o $out_dir/$src_name.s
printf "$fmt_start_nl" "Compiled using:" "LLC"

# Assemble
$clangPP -c -g -$opt_lvl $out_dir/$src_name.s -o $out_dir/$src_name.o
printf "$fmt_list" "Assembled using:" "Clang++"

# Link
$clangPP $driver $out_dir/$src_name.o \
  -g -$opt_lvl -o $out_dir/$src_name.out -DUSE_MPI=0 -DUSE_EXTERNAL_$src_name
printf "$fmt_list" "Linked with:" $driver

# Execute
printf "$fmt_start_nl" "Executing:" "$out_dir/$src_name.out"
res=$(./$out_dir/$src_name.out)
printf "\n%s\n" "$res"

# Check for correctness
expected=$(grep -A8 "Run completed" normal_output.txt)
actual=$(echo "$res" | grep -A8 "Run completed")

if [[ "$actual" == "$expected" ]]; then
printf "$fmt_start_nl" "Output correct"
else
printf "$fmt_err" "Wrong output"
fi
