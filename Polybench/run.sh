#!/bin/bash

# Usage: ./run.sh <benchmark>

# Settings
util_folder=./benchmarks/utilities
driver=./benchmarks/utilities/polybench.c
flags="-DSMALL_DATASET -DPOLYBENCH_TIME"
opt_lvl=O0
out_dir=./out

clang=$(which clang)                     || clang="NOT FOUND"
clangPP=$(which clang++)                 || clangPP="NOT FOUND"
mlir_opt=$(which mlir-opt)               || mlir_opt="NOT FOUND"
cf_opt=$(which cf-opt)                   || cf_opt="NOT FOUND"
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
printf "$fmt_list" "cf-opt:" $cf_opt
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
   [ "$cf_opt" == "NOT FOUND" ] || \
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

# Find mangled name
$compiler -c -I $util_folder $src -o $out_dir/$src_name.o
mangled_name=$(nm $out_dir/$src_name.o | grep $src_name | cut -d ' ' -f3)
rm $out_dir/$src_name.o
printf "$fmt_list\n" "Function name:" "$mangled_name"

# Interface renaming function
rename_interface(){
  sed -i -e "s/_mlir_ciface_$1/_mlir_ciface_tmp/g" $out_dir/$src_name.mlir
  sed -i -e "s/$1/$1_renamed/g" $out_dir/$src_name.mlir
  sed -i -e "s/_mlir_ciface_tmp/$1/g" $out_dir/$src_name.mlir
  printf "$fmt_list" "$1"
}

# Generate straight translation
$cgeist -resource-dir=$(clang -print-resource-dir) -I $util_folder \
  -function=$mangled_name -S --memref-fullrank -$opt_lvl --raise-scf-to-affine \
  $src > $out_dir/$src_name\_cgeist_nopt.mlir
printf "$fmt_start" "Generated:" "non-opt-version"

# Polygeist c/cpp -> mlir
$cgeist -resource-dir=$(clang -print-resource-dir) -I $util_folder \
  -S --memref-fullrank -$opt_lvl --raise-scf-to-affine $flags \
  --memref-abi=0 --struct-abi=0 $src > $out_dir/$src_name.mlir
cp $out_dir/$src_name.mlir $out_dir/$src_name\_cgeist.mlir
printf "$fmt_list" "Lowered to:" "MLIR"

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
  # Workaround for https://github.com/llvm/llvm-project/issues/55301
  $cf_opt --cf-index-to-int --allow-unregistered-dialect \
      $out_dir/$src_name.mlir > $out_dir/$src_name\_cf_fix.mlir
  cp $out_dir/$src_name\_cf_fix.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Applied:" "Workaround"

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
if grep -q -i "_mlir_ciface_" $out_dir/$src_name.mlir; then
  printf "$fmt_start_nl" "Renamed Interfaces:"

  while grep -i "_mlir_ciface_" $out_dir/$src_name.mlir > /dev/null; do
    line=$(grep -i "func @_mlir_ciface_" $out_dir/$src_name.mlir | head -1 )
    if_name=$(echo $line | sed 's/.*_mlir_ciface_\(.*\)(.*/\1/')
    rename_interface $if_name
  done
fi

# Translate
$mlir_translate --mlir-to-llvmir $out_dir/$src_name.mlir > $out_dir/$src_name.ll
printf "$fmt_start_nl" "Translated to:" "LLVMIR"

# Compile
$llc -$opt_lvl --relocation-model=pic $out_dir/$src_name.ll -o $out_dir/$src_name.s
printf "$fmt_list" "Compiled using:" "LLC"

# Assemble
$compiler -c -g -$opt_lvl $out_dir/$src_name.s -o $out_dir/$src_name.o
printf "$fmt_list" "Assembled using:" "$(basename ${compiler%.*})"

# Link
$compiler $driver $out_dir/$src_name.o $flags \
  -g -$opt_lvl -o $out_dir/$src_name.out
printf "$fmt_list" "Linked with:" $driver

# Execute
printf "$fmt_start_nl" "Executing:" "$out_dir/$src_name.out"
res=$(./$out_dir/$src_name.out)
printf "\n%s\n" "$res"
