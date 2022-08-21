#!/bin/bash

# Usage: ./run.sh <cpp file>

# Exactly one global text symbol needed in the source file

# Settings
driver=lulesh_driver.cpp
flags="-DUSE_MPI=0"
opt_lvl=-O3
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
printf "$fmt_list" "opt lvl:" $opt_lvl

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

# Generate executables
# $gcc $opt_lvl $flags -o $out_dir/$src_name\_gcc.out $src $driver -lm
# printf "$fmt_list" "Generated:" "GCC"
$gpp $opt_lvl $flags -o $out_dir/$src_name\_gpp.out $src $driver
printf "$fmt_list" "Generated:" "G++"
# $clang  $opt_lvl $flags -o $out_dir/$src_name\_clang.out $src $driver -lm
# printf "$fmt_list" "Generated:" "Clang"
$clangpp $opt_lvl $flags -o $out_dir/$src_name\_clangpp.out $src $driver
printf "$fmt_list" "Generated:" "Clang++"

# Find mangled name
$clangpp -c $src $flags -o $out_dir/$src_name.o
mangled_name=$(nm $out_dir/$src_name.o | grep $src_name | cut -d ' ' -f3)
rm $out_dir/$src_name.o
printf "$fmt_list\n" "Function name:" "$mangled_name"

# Generate straight translation
$cgeist -resource-dir=$($clang -print-resource-dir) -function=$mangled_name \
  -S --memref-fullrank -O0 $flags $src | \
$mlir_opt --allow-unregistered-dialect --lower-affine \
  > $out_dir/$src_name\_noopt.mlir
printf "$fmt_start" "Generated:" "Non-optimized MLIR"

# Polygeist c/cpp -> mlir
$cgeist -resource-dir=$($clang -print-resource-dir) -function=$mangled_name \
  -S --memref-fullrank --memref-abi=0 $opt_lvl $flags \
  --raise-scf-to-affine $src | \
# $mlir_opt --allow-unregistered-dialect --affine-loop-invariant-code-motion | \
# $mlir_opt --allow-unregistered-dialect --affine-scalrep | \
$mlir_opt --allow-unregistered-dialect --lower-affine | \
$mlir_opt --allow-unregistered-dialect --cse --inline \
  > $out_dir/$src_name\_opt.mlir
cp $out_dir/$src_name\_opt.mlir $out_dir/$src_name.mlir
printf "$fmt_list" "Generated:" "Optimized MLIR"

# Generate memref version
$cgeist -resource-dir=$($clang -print-resource-dir) -function=$mangled_name \
  -S --memref-fullrank $opt_lvl $flags \
  --raise-scf-to-affine $src | \
$mlir_opt --allow-unregistered-dialect --affine-loop-invariant-code-motion | \
$mlir_opt --allow-unregistered-dialect --affine-scalrep | \
$mlir_opt --allow-unregistered-dialect --lower-affine | \
$mlir_opt --allow-unregistered-dialect --cse --inline \
  > $out_dir/$src_name\_opt_mem.mlir
printf "$fmt_list" "Generated:" "Optimized MLIR (Memref)"

# Lower Polygeist
if grep -q -i "polygeist" $out_dir/$src_name.mlir; then
  $polygeist_opt --convert-polygeist-to-llvm \
    $out_dir/$src_name.mlir > $out_dir/$src_name\_polygeist.mlir
  cp $out_dir/$src_name\_polygeist.mlir $out_dir/$src_name.mlir
  printf "$fmt_list" "Lowered:" "Polygeist"
fi

# Lower to llvm 
$mlir_opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
  --convert-math-to-llvm --lower-host-to-llvm --reconcile-unrealized-casts \
  $out_dir/$src_name.mlir > $out_dir/$src_name\_llvm.mlir
cp $out_dir/$src_name\_llvm.mlir $out_dir/$src_name.mlir
printf "$fmt_list" "Lowered to:" "LLVM"

# Interface renaming function
rename_interface(){
  sed -i -e "s/_mlir_ciface_$1/_mlir_ciface_tmp/g" $out_dir/$src_name.mlir
  sed -i -e "s/$1/$1_renamed/g" $out_dir/$src_name.mlir
  sed -i -e "s/_mlir_ciface_tmp/$1/g" $out_dir/$src_name.mlir
  printf "$fmt_list" "$1"
}

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
$llc $opt_lvl --relocation-model=pic $out_dir/$src_name.ll -o $out_dir/$src_name.s
printf "$fmt_list" "Compiled using:" "LLC"

# Assemble
$clangpp -c -g $opt_lvl $out_dir/$src_name.s -o $out_dir/$src_name.o
printf "$fmt_list" "Assembled using:" "Clang++"

# Link
$clangpp $driver $out_dir/$src_name.o \
  -g $opt_lvl -o $out_dir/$src_name.out $flags -DUSE_EXTERNAL_$src_name
printf "$fmt_list" "Linked with:" $driver

# Compile SDFG
$sdfg_opt --convert-to-sdfg $out_dir/$src_name\_opt.mlir \
| $sdfg_translate --mlir-to-sdfg | $python opt.py $out_dir/$src_name\_opt.sdfg
printf "$fmt_list" "Compiled:" "Optimized SDFG"

$sdfg_opt --convert-to-sdfg $out_dir/$src_name\_noopt.mlir \
| $sdfg_translate --mlir-to-sdfg | $python opt.py $out_dir/$src_name\_noopt.sdfg
printf "$fmt_list" "Compiled:" "Non-Optimized SDFG"

# Run benchmark
timings=$out_dir/timings.txt
touch $timings

printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_start_nl" "Running:" "G++"
echo -e "\n--- G++ ---" >> $timings
for i in $(seq 1 $repetitions); do
  ts=$(date +%s%N)
  ./$out_dir/$src_name\_gpp.out
  echo $((($(date +%s%N) - $ts)/1000000)) >> $timings
done

printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_start_nl" "Running:" "Clang++"
echo -e "\n--- Clang++ ---" >> $timings
for i in $(seq 1 $repetitions); do
  ts=$(date +%s%N)
  ./$out_dir/$src_name\_clangpp.out
  echo $((($(date +%s%N) - $ts)/1000000)) >> $timings
done

printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_start_nl" "Running:" "MLIR"
echo -e "\n--- MLIR ---" >> $timings
for i in $(seq 1 $repetitions); do
  ts=$(date +%s%N)
  ./$out_dir/$src_name\_mlir.out
  echo $((($(date +%s%N) - $ts)/1000000)) >> $timings
done

printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_start_nl" "Running:" "SDFG Opt"
echo -e "\n--- SDFG OPT ---" >> $timings
for i in $(seq 1 $repetitions); do
  $python run.py $out_dir/$src_name\_opt.sdfg >> $timings
done

printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_start_nl" "Running:" "SDFG Non-Opt"
echo -e "\n--- SDFG NOOPT ---" >> $timings
for i in $(seq 1 $repetitions); do
  $python run.py $out_dir/$src_name\_noopt.sdfg >> $timings
done

