#!/bin/bash

# Usage: ./run.sh <cpp file>

# Exactly one global text symbol needed in the source file

# Settings
driver=lulesh_driver.cpp
flags="-O3"
out_dir=./out

gcc=$(which gcc)                         || gcc="NOT FOUND"
gpp=$(which g++)                         || gpp="NOT FOUND"
clang=$(which clang)                     || clang="NOT FOUND"
clangpp=$(which clang++)                 || clangpp="NOT FOUND"
cgeist=$(which cgeist)                   || cgeist="NOT FOUND"
polygeist_opt=$(which polygeist-opt)     || polygeist_opt="NOT FOUND"
mlir_opt=$(which mlir-opt)               || mlir_opt="NOT FOUND"
mlir_translate=$(which mlir-translate)   || mlir_translate="NOT FOUND"
sdfg_opt=$(which sdfg-opt)               || sdfg_opt="NOT FOUND"
sdfg_translate=$(which sdfg-translate)   || sdfg_translate="NOT FOUND"
python=$(which python3)                  || python="NOT FOUND"
llc=$(which llc)                         || llc="NOT FOUND"

fmt_start="🔥 %-18s %s\n"
fmt_start_nl="\n🔥 %-18s %s\n"
fmt_list="   %-18s %s\n"
fmt_err="\n❌ %s\n"

printf "$fmt_start" "gcc:" $gcc
printf "$fmt_list" "g++:" $gpp
printf "$fmt_list" "clang:" $clang
printf "$fmt_list" "clang++:" $clangpp
printf "$fmt_list" "cgeist:" $cgeist
printf "$fmt_list" "polygeist-opt:" $polygeist_opt
printf "$fmt_list" "mlir-opt:" $mlir_opt
printf "$fmt_list" "mlir-translate:" $mlir_translate
printf "$fmt_list" "sdfg-opt:" $sdfg_opt
printf "$fmt_list" "sdfg-translate:" $sdfg_translate
printf "$fmt_list" "python:" $python
printf "$fmt_list" "llc:" $llc
printf "$fmt_list" "output dir:" $out_dir
printf "$fmt_list" "flags:" $flags

# Check if tools exist
if [ "$gcc" == "NOT FOUND" ] || \
   [ "$gpp" == "NOT FOUND" ] || \
   [ "$clang" == "NOT FOUND" ] || \
   [ "$clangpp" == "NOT FOUND" ] || \
   [ "$cgeist" == "NOT FOUND" ] || \
   [ "$polygeist_opt" == "NOT FOUND" ]|| \
   [ "$mlir_opt" == "NOT FOUND" ] || \
   [ "$mlir_translate" == "NOT FOUND" ] || \
   [ "$sdfg_opt" == "NOT FOUND" ] || \
   [ "$sdfg_translate" == "NOT FOUND" ] || \
   [ "$python" == "NOT FOUND" ] || \
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
if [ -z ${1+x} ]; then
  printf "$fmt_err" "Please provide a C/C++ file"
  exit 1;
fi
src=$1
src_name=$(basename ${src%.*})
src_ext=${src##*.}
src_dir=$(dirname $src)
printf "$fmt_start_nl" "Source:" "$src_name ($src)"

# Find mangled name
$clangpp -c $src -O0 -o $out_dir/$src_name.o
mangled_name=$(nm $out_dir/$src_name.o | grep " T " | cut -d ' ' -f3)
rm $out_dir/$src_name.o
printf "$fmt_list\n" "Function name:" "$mangled_name"

# Generate straight translation
$cgeist -resource-dir=$($clang -print-resource-dir) -function=$mangled_name \
  -S --memref-fullrank -O3 --raise-scf-to-affine $src | \
$mlir_opt --affine-loop-invariant-code-motion | $mlir_opt --affine-scalrep | \
$mlir_opt --lower-affine | $mlir_opt --cse > $out_dir/$src_name.mlir
printf "$fmt_start" "Generated:" "MLIR"

# Compile SDFG
# sed -i -e "s/_Z6sftrfri/main/g" $out_dir/$src_name.mlir
$sdfg_opt --convert-to-sdfg $out_dir/$src_name.mlir \
| $sdfg_translate --mlir-to-sdfg | $python opt.py $out_dir/$src_name\_opt.sdfg
printf "$fmt_list" "Compiled:" "Optimized SDFG"

exit 0

# Run benchmark
timings=$out_dir/timings.txt
touch $timings

printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_start_nl" "Running:" "SDFG Opt"
echo -e "\n--- SDFG OPT ---" >> $timings
for i in $(seq 1 $repetitions); do
  $python run.py $out_dir/$src_name\_opt.sdfg >> $timings
done
