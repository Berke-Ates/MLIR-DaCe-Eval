#!/bin/bash

# Usage: ./run.sh <benchmark> <repetitions>

# Settings
util_folder=./benchmarks/utilities
driver=./benchmarks/utilities/polybench.c
flags="-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -fPIC -march=native"
opt_lvl=-O3
out_dir=./out

export DACE_compiler_cpu_executable="$(which clang++)"
export CC=`which clang`
export CXX=`which clang++`
export DACE_compiler_cpu_openmp_sections=0
export DACE_instrumentation_report_each_invocation=0
export DACE_compiler_cpu_args="-fPIC -O3 -march=native"
# export DACE_debugprint=verbose

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
fmt_file="%s\n"
fmt_file_nl="\n%s\n"

# printf "$fmt_start" "gcc:" $gcc
# printf "$fmt_list" "g++:" $gpp
# printf "$fmt_list" "clang:" $clang
# printf "$fmt_list" "clang++:" $clangpp
# printf "$fmt_list" "cgeist:" $cgeist
# printf "$fmt_list" "polygeist-opt:" $polygeist_opt
# printf "$fmt_list" "mlir-opt:" $mlir_opt
# printf "$fmt_list" "mlir-translate:" $mlir_translate
# printf "$fmt_list" "sdfg-opt:" $sdfg_opt
# printf "$fmt_list" "sdfg-translate:" $sdfg_translate
# printf "$fmt_list" "python:" $python
# printf "$fmt_list" "llc:" $llc
# printf "$fmt_list" "output dir:" $out_dir
# printf "$fmt_list" "opt lvl:" $opt_lvl
   
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

# Runs a compiler through all benchmarks
compile_all(){
  benchmarks=$(find benchmarks/* -name '*.c' -not -path "benchmarks/utilities/*")
  for filename in $benchmarks; do
    $1 $filename &> /dev/null
  done
}

### GCC ###
printf "$fmt_file" "--- GCC ---"

compile_with_gcc(){
  src=$1
  src_name=$(basename ${src%.*})
  $gcc -I $util_folder $opt_lvl $flags -DPOLYBENCH_TIME -o $out_dir/$src_name\_gcc.out $src $driver -lm
}

time compile_all compile_with_gcc

### Clang ###
printf "$fmt_file_nl" "--- Clang ---"

compile_with_clang(){
  src=$1
  src_name=$(basename ${src%.*})
  $clang -I $util_folder $opt_lvl $flags -DPOLYBENCH_TIME -o $out_dir/$src_name\_clang.out $src $driver -lm
}

time compile_all compile_with_clang

### MLIR ###
printf "$fmt_file_nl" "--- MLIR ---"

compile_with_mlir(){
  src=$1
  src_name=$(basename ${src%.*})

  $cgeist -resource-dir=$($clang -print-resource-dir) -I $util_folder \
    -S --memref-fullrank $opt_lvl --raise-scf-to-affine $flags -DPOLYBENCH_TIME $src | \
  $mlir_opt --affine-loop-invariant-code-motion | \
  $mlir_opt --affine-scalrep | \
  $mlir_opt --lower-affine | \
  $mlir_opt --cse --inline \
    > $out_dir/$src_name\_opt.mlir

  $mlir_opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
    --convert-math-to-llvm --lower-host-to-llvm --reconcile-unrealized-casts \
    $out_dir/$src_name\_opt.mlir > $out_dir/$src_name.mlir

  $mlir_translate --mlir-to-llvmir $out_dir/$src_name.mlir > $out_dir/$src_name.ll
  $llc $opt_lvl --relocation-model=pic $out_dir/$src_name.ll -o $out_dir/$src_name.s
  $clang $opt_lvl $flags -DPOLYBENCH_TIME  $out_dir/$src_name.s $driver -o $out_dir/$src_name\_mlir.out -lm
}

time compile_all compile_with_mlir

### SDFG ###
printf "$fmt_file_nl" "--- SDFG ---"

compile_with_sdfg(){
  src=$1
  src_name=$(basename ${src%.*})

  $cgeist -resource-dir=$($clang -print-resource-dir) -I $util_folder \
    -S --memref-fullrank $opt_lvl --raise-scf-to-affine $flags \
    -DPOLYBENCH_DUMP_ARRAYS $src | \
  $mlir_opt --affine-loop-invariant-code-motion | \
  $mlir_opt --affine-scalrep | \
  $mlir_opt --lower-affine | \
  $mlir_opt --cse --inline \
    > $out_dir/$src_name\_opt_sdfg.mlir

  $sdfg_opt --convert-to-sdfg $out_dir/$src_name\_opt_sdfg.mlir | \
  $sdfg_translate --mlir-to-sdfg |\
  $python opt_compile.py $out_dir/$src_name\_opt.sdfg
}

time compile_all compile_with_sdfg

### SDFG No Python ###
printf "$fmt_file_nl" "--- SDFG No Python ---"

compile_with_sdfg(){
  src=$1
  src_name=$(basename ${src%.*})

  $cgeist -resource-dir=$($clang -print-resource-dir) -I $util_folder \
    -S --memref-fullrank $opt_lvl --raise-scf-to-affine $flags \
    -DPOLYBENCH_DUMP_ARRAYS $src | \
  $mlir_opt --affine-loop-invariant-code-motion | \
  $mlir_opt --affine-scalrep | \
  $mlir_opt --lower-affine | \
  $mlir_opt --cse --inline \
    > $out_dir/$src_name\_opt_sdfg.mlir

  $sdfg_opt --convert-to-sdfg $out_dir/$src_name\_opt_sdfg.mlir | \
  $sdfg_translate --mlir-to-sdfg 
}

time compile_all compile_with_sdfg
