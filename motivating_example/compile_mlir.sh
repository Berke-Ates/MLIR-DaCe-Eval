#!/bin/bash

# Usage: ./compile_mlir.sh <mlir file>

# Settings
opt_lvl=O0
out_dir=./out

clang=$(which clang)                     || clang="NOT FOUND"
mlir_opt=$(which mlir-opt)               || mlir_opt="NOT FOUND"
cf_opt=$(which cf-opt)                   || cf_opt="NOT FOUND"
mlir_translate=$(which mlir-translate)   || mlir_translate="NOT FOUND"
llc=$(which llc)                         || llc="NOT FOUND"

fmt_start="🔥 %-18s %s\n"
fmt_start_nl="\n🔥 %-18s %s\n"
fmt_list="   %-18s %s\n"
fmt_err="\n❌ %s\n"

printf "$fmt_list" "clang:" $clang
printf "$fmt_list" "mlir-opt:" $mlir_opt
printf "$fmt_list" "cf-opt:" $cf_opt
printf "$fmt_list" "llc:" $llc
printf "$fmt_list" "output dir:" $out_dir
printf "$fmt_list" "opt lvl:" $opt_lvl


set -e # Fail fast

# Check if tools exist
if [ "$clang" == "NOT FOUND" ] || \
   [ "$mlir_opt" == "NOT FOUND" ] || \
   [ "$cf_opt" == "NOT FOUND" ] || \
   [ "$mlir_translate" == "NOT FOUND" ] || \
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

# Copy mlir
cp $src $out_dir/$src_name.mlir

# Lower affine 
if grep -q -i "affine" $out_dir/$src_name.mlir; then
  $mlir_opt --lower-affine \
    $out_dir/$src_name.mlir > $out_dir/$src_name\_affine.mlir
  cp $out_dir/$src_name\_affine.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Lowered:" "Affine"
fi

# Lower scf 
if grep -q -i "scf" $out_dir/$src_name.mlir; then
  $mlir_opt --convert-scf-to-cf \
     $out_dir/$src_name.mlir > $out_dir/$src_name\_scf.mlir
  cp $out_dir/$src_name\_scf.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Lowered:" "SCF"
fi

# Lower cf
if grep -q -i "cf" $out_dir/$src_name.mlir; then
  # Workaround for https://github.com/llvm/llvm-project/issues/55301
  $cf_opt --cf-index-to-int \
      $out_dir/$src_name.mlir > $out_dir/$src_name\_cf_fix.mlir
  cp $out_dir/$src_name\_cf_fix.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Applied Workaround"

  $mlir_opt --convert-cf-to-llvm \
     $out_dir/$src_name.mlir > $out_dir/$src_name\_cf.mlir
  cp $out_dir/$src_name\_cf.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Lowered:" "CF"
fi

# Lower to llvm 
$mlir_opt --lower-host-to-llvm $out_dir/$src_name.mlir > $out_dir/$src_name\_llvm.mlir
cp $out_dir/$src_name\_llvm.mlir $out_dir/$src_name.mlir
printf "$fmt_list" "Lowered to:" "LLVM"

# Translate
$mlir_translate --mlir-to-llvmir $out_dir/$src_name.mlir > $out_dir/$src_name.ll
printf "$fmt_list" "Translated to:" "LLVMIR"

# Compile
$llc -$opt_lvl --relocation-model=pic $out_dir/$src_name.ll -o $out_dir/$src_name.s
printf "$fmt_start_nl" "Compiled using:" "LLC"

# Assemble
$clang -$opt_lvl $out_dir/$src_name.s -o $out_dir/$src_name.out
printf "$fmt_list" "Assembled using:" "Clang"
