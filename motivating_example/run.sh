#!/bin/bash

# Usage: ./run.sh <cpp file>

# Settings
flags="-fPIC -march=native"
opt_lvl=-O3
out_dir=./out
max_time=1m
repetitions=10
gc_time=10

export DACE_compiler_cpu_executable="$(which clang++)"
export CC=`which clang`
export CXX=`which clang++`
export DACE_compiler_cpu_openmp_sections=0
export DACE_instrumentation_report_each_invocation=0
export DACE_compiler_cpu_args="-fPIC -O3 -march=native"

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
src="loop.c"
src_name=$(basename ${src%.*})
src_ext=${src##*.}
src_dir=$(dirname $src)
src_chrono="loop_chrono.c"
printf "$fmt_start_nl" "Source:" "$src_name ($src)"

# Generate executables
$gcc $opt_lvl $flags -o $out_dir/$src_name\_gcc.out $src_chrono
printf "$fmt_list" "Generated:" "GCC"
$clang $opt_lvl $flags -o $out_dir/$src_name\_clang.out $src_chrono
printf "$fmt_list" "Generated:" "Clang"

# Generate mlir
$cgeist -resource-dir=$($clang -print-resource-dir) \
  -S --memref-fullrank $opt_lvl --raise-scf-to-affine $src | \
$mlir_opt --affine-loop-invariant-code-motion | $mlir_opt --affine-scalrep | \
$mlir_opt --lower-affine | $mlir_opt --cse > $out_dir/$src_name\_opt.mlir
printf "$fmt_list" "Generated:" "Optimized MLIR"

$cgeist -resource-dir=$($clang -print-resource-dir) \
  -S --memref-fullrank -O0 $src | \
$mlir_opt --lower-affine > $out_dir/$src_name\_noopt.mlir
printf "$fmt_list" "Generated:" "Non-optimized MLIR"

# Generate mlir
$cgeist $flags -resource-dir=$($clang -print-resource-dir) \
  -S --memref-fullrank $opt_lvl --raise-scf-to-affine $src_chrono | \
$mlir_opt --affine-loop-invariant-code-motion | $mlir_opt --affine-scalrep | \
$mlir_opt --lower-affine | $mlir_opt --cse > $out_dir/$src_name\_cgeist.mlir
printf "$fmt_list" "Generated:" "Polygeist MLIR"

### Lower through MLIR ###

# Lower to llvm 
$mlir_opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
  --lower-host-to-llvm --reconcile-unrealized-casts \
  $out_dir/$src_name\_cgeist.mlir > $out_dir/$src_name.mlir
printf "$fmt_list" "Lowered to:" "LLVM"

# Translate
$mlir_translate --mlir-to-llvmir $out_dir/$src_name.mlir > $out_dir/$src_name.ll
printf "$fmt_list" "Translated to:" "LLVMIR"

# Compile
$llc $opt_lvl $out_dir/$src_name.ll -o $out_dir/$src_name.s
printf "$fmt_list" "Compiled using:" "LLC"

# Assemble
$clang -no-pie $opt_lvl $flags $out_dir/$src_name.s -o $out_dir/$src_name\_mlir.out
printf "$fmt_list" "Assembled using:" "Clang"

# Compile SDFG
$sdfg_opt --convert-to-sdfg $out_dir/$src_name\_opt.mlir \
| $sdfg_translate --mlir-to-sdfg | $python opt.py $out_dir/$src_name\_opt.sdfg
printf "$fmt_list" "Compiled:" "Optimized SDFG"

$sdfg_opt --convert-to-sdfg $out_dir/$src_name\_noopt.mlir \
| $sdfg_translate --mlir-to-sdfg | $python opt.py $out_dir/$src_name\_noopt.sdfg
printf "$fmt_list" "Compiled:" "Non-Optimized SDFG"

cat loop-c2dace.sdfg | $python opt_noauto.py $out_dir/$src_name\_c2dace.sdfg
printf "$fmt_list" "Compiled:" "Non-Optimized SDFG"

# Run benchmark
timings=$out_dir/timings.txt
touch $timings

# printf "$fmt_list" "Waiting for GC"
# sleep $gc_time

# printf "$fmt_start_nl" "Running:" "GCC"
# echo "--- GCC ---" >> $timings
# for i in $(seq 1 $repetitions); do
#   $out_dir/$src_name\_gcc.out >> $timings
# done


# printf "$fmt_list" "Waiting for GC"
# sleep $gc_time

# printf "$fmt_list" "Running:" "Clang"
# echo -e "\n--- Clang ---" >> $timings
# for i in $(seq 1 $repetitions); do
#   $out_dir/$src_name\_clang.out >> $timings
# done

# printf "$fmt_list" "Waiting for GC"
# sleep $gc_time

# printf "$fmt_list" "Running:" "SDFG Non-Opt"
# echo -e "\n--- SDFG NOOPT ---" >> $timings
# $python run.py $out_dir/$src_name\_noopt.sdfg $repetitions
# $python eval.py >> $timings

# printf "$fmt_list" "Waiting for GC"
# sleep $gc_time

# printf "$fmt_list" "Running:" "MLIR"
# echo -e "\n--- MLIR ---" >> $timings
# for i in $(seq 1 $repetitions); do
#   $out_dir/$src_name\_mlir.out >> $timings
# done

# printf "$fmt_list" "Waiting for GC"
# sleep $gc_time

# printf "$fmt_list" "Running:" "SDFG Opt"
# echo -e "\n--- SDFG OPT ---" >> $timings
# $python run.py $out_dir/$src_name\_opt.sdfg $repetitions
# $python eval.py >> $timings


printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_list" "Running:" "C2DaCe"
echo -e "\n--- C2DaCe ---" >> $timings
$python run_c2dace.py $out_dir/$src_name\_c2dace.sdfg $repetitions
$python eval.py >> $timings
