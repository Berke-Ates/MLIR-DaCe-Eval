#!/bin/bash

# Usage: ./run.sh <c file>

### Settings ###
flags="-fPIC -O2 -march=native"
out_dir=./out
repetitions=100
gc_time=10

export DACE_compiler_cpu_openmp_sections=0
export DACE_compiler_cpu_args="$flags"
export DACE_instrumentation_report_each_invocation=0
# export DACE_debugprint=verbose

gcc=$(which gcc)                         || gcc="NOT FOUND"
clang=$(which clang)                     || clang="NOT FOUND"
cgeist=$(which cgeist)                   || cgeist="NOT FOUND"
polygeist_opt=$(which polygeist-opt)     || polygeist_opt="NOT FOUND"
mlir_opt=$(which mlir-opt)               || mlir_opt="NOT FOUND"
mlir_translate=$(which mlir-translate)   || mlir_translate="NOT FOUND"
sdfg_opt=$(which sdfg-opt)               || sdfg_opt="NOT FOUND"
sdfg_translate=$(which sdfg-translate)   || sdfg_translate="NOT FOUND"
python=$(which python3)                  || python="NOT FOUND"
llc=$(which llc)                         || llc="NOT FOUND"

### Formats & Info ###
fmt_start="🔥 %-18s %s\n"
fmt_start_nl="\n🔥 %-18s %s\n"
fmt_list="   %-18s %s\n"
fmt_err="\n❌ %s\n"

printf "$fmt_start" "gcc:" $gcc
printf "$fmt_list" "clang:" $clang
printf "$fmt_list" "cgeist:" $cgeist
printf "$fmt_list" "polygeist-opt:" $polygeist_opt
printf "$fmt_list" "mlir-opt:" $mlir_opt
printf "$fmt_list" "mlir-translate:" $mlir_translate
printf "$fmt_list" "sdfg-opt:" $sdfg_opt
printf "$fmt_list" "sdfg-translate:" $sdfg_translate
printf "$fmt_list" "python:" $python
printf "$fmt_list" "llc:" $llc
printf "$fmt_list" "output dir:" $out_dir
printf "$fmt_list" "flags:" "$flags"

### Check if tools exist ###
if [ "$gcc" == "NOT FOUND" ] || \
   [ "$clang" == "NOT FOUND" ] || \
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

### Create output directory ###
if [ ! -d $out_dir ]; then
  mkdir -p $out_dir;
fi
rm -rf $out_dir/* # Clear output directory

### Get filename ###
if [ -z ${1+x} ]; then
  printf "$fmt_err" "Please provide a C file"
  exit 1;
fi
src=$1
src_name=$(basename ${src%.*})
src_ext=${src##*.}
src_dir=$(dirname $src)
src_chrono="$src_dir/$src_name\_chrono.c"
printf "$fmt_start_nl" "Source:" "$src_name ($src)"

### Generate executables ###
$gcc $opt_lvl $flags -o $out_dir/$src_name\_gcc.out $src_chrono -lm
printf "$fmt_list" "Generated:" "GCC"
$clang $opt_lvl $flags -o $out_dir/$src_name\_clang.out $src_chrono -lm
printf "$fmt_list" "Generated:" "Clang"

#### Generate mlir ###
$cgeist -resource-dir=$($clang -print-resource-dir) \
  -S --memref-fullrank $opt_lvl --raise-scf-to-affine $src_chrono | \
$mlir_opt --affine-loop-invariant-code-motion -allow-unregistered-dialect | \
$mlir_opt --affine-scalrep -allow-unregistered-dialect | \
$mlir_opt --lower-affine -allow-unregistered-dialect | \
$mlir_opt --cse -allow-unregistered-dialect > $out_dir/$src_name\_chrono.mlir
cp $out_dir/$src_name\_chrono.mlir $out_dir/$src_name.mlir
printf "$fmt_list" "Generated:" "Optimized MLIR (Chrono)"

$cgeist -resource-dir=$($clang -print-resource-dir) \
  -S --memref-fullrank $opt_lvl --raise-scf-to-affine $src | \
$mlir_opt --affine-loop-invariant-code-motion -allow-unregistered-dialect | \
$mlir_opt --affine-scalrep -allow-unregistered-dialect | \
$mlir_opt --lower-affine -allow-unregistered-dialect | \
$mlir_opt --cse -allow-unregistered-dialect > $out_dir/$src_name\_opt.mlir
printf "$fmt_list" "Generated:" "Optimized MLIR"

### Lower through MLIR ###

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

# Translate
$mlir_translate --mlir-to-llvmir $out_dir/$src_name.mlir > $out_dir/$src_name.ll
printf "$fmt_list" "Translated to:" "LLVMIR"

# Compile
$llc $opt_lvl $out_dir/$src_name.ll -o $out_dir/$src_name.s
printf "$fmt_list" "Compiled using:" "LLC"

# Assemble
$clang -no-pie $opt_lvl $flags $out_dir/$src_name.s -o $out_dir/$src_name\_mlir.out -lm
printf "$fmt_list" "Assembled using:" "Clang"

# Compile SDFG
$sdfg_opt --convert-to-sdfg $out_dir/$src_name\_opt.mlir \
| $sdfg_translate --mlir-to-sdfg | $python opt.py $out_dir/$src_name\_opt.sdfg
printf "$fmt_list" "Compiled:" "Optimized SDFG"

### Run benchmark ###
timings=$out_dir/timings.txt
touch $timings

# GCC
printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_start_nl" "Running:" "GCC"
echo "--- GCC ---" >> $timings
for i in $(seq 1 $repetitions); do
  $out_dir/$src_name\_gcc.out >> $timings
done

# Clang
printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_list" "Running:" "Clang"
echo -e "\n--- Clang ---" >> $timings
for i in $(seq 1 $repetitions); do
  $out_dir/$src_name\_clang.out >> $timings
done

# MLIR
printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_list" "Running:" "MLIR"
echo -e "\n--- MLIR ---" >> $timings
for i in $(seq 1 $repetitions); do
  $out_dir/$src_name\_mlir.out >> $timings
done

# SDFG OPT
printf "$fmt_list" "Waiting for GC"
sleep $gc_time

printf "$fmt_list" "Running:" "SDFG Opt"
echo -e "\n--- SDFG OPT ---" >> $timings
$python run.py $out_dir/$src_name\_opt.sdfg
$python eval.py >> $timings
