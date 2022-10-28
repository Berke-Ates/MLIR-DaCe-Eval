#!/bin/bash

# Settings
util_folder=./benchmarks/utilities
driver=./benchmarks/utilities/polybench.c
flags="-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -fPIC -march=native"
opt_lvl=-O3
out_dir=./compile_times

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

save_time(){
  { time $1 2>&1 ; } 2> $2
}

compile_with_gcc(){
  $gcc -I $util_folder $opt_lvl $flags -DPOLYBENCH_TIME -o /dev/null $1 \
  $driver -lm &> /dev/null
}

compile_with_clang(){
  $clang -I $util_folder $opt_lvl $flags -DPOLYBENCH_TIME -o /dev/null $1 \
  $driver -lm &> /dev/null
}

compile_with_mlir_cgeist(){
  $cgeist -resource-dir=$($clang -print-resource-dir) -I $util_folder \
  -S --memref-fullrank $opt_lvl --raise-scf-to-affine $flags -DPOLYBENCH_TIME \
  $1 > $2 
}

compile_with_mlir_lowering(){
  $mlir_opt --affine-loop-invariant-code-motion $1 | \
  $mlir_opt --affine-scalrep | \
  $mlir_opt --lower-affine | \
  $mlir_opt --cse --inline | \
  $mlir_opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
    --convert-math-to-llvm --lower-host-to-llvm --reconcile-unrealized-casts | \
  $mlir_translate --mlir-to-llvmir > $2
}

compile_with_mlir_llc(){
  $llc $opt_lvl --relocation-model=pic $1 -o $2
}

compile_with_mlir_clang(){
  $clang $opt_lvl $flags -DPOLYBENCH_TIME $1 $driver -o /dev/null -lm
}

compile_with_mlir(){
  src=$1
  src_name=$(basename ${src%.*})
  mkdir tmp_folder
  
  save_time "compile_with_mlir_cgeist $src tmp_folder/cgeist.mlir" \
    $out_dir/$src_name/mlir/cgeist.txt
  save_time "compile_with_mlir_lowering tmp_folder/cgeist.mlir tmp_folder/lowered.ll" \
    $out_dir/$src_name/mlir/lowering.txt
  save_time "compile_with_mlir_llc tmp_folder/lowered.ll tmp_folder/compiled.s" \
    $out_dir/$src_name/mlir/llc.txt
  save_time "compile_with_mlir_clang tmp_folder/compiled.s" \
    $out_dir/$src_name/mlir/clang.txt

  rm -r tmp_folder
}

compile_with_dcir_cgeist(){
  $cgeist -resource-dir=$($clang -print-resource-dir) -I $util_folder \
    -S --memref-fullrank $opt_lvl --raise-scf-to-affine $flags \
    -DPOLYBENCH_DUMP_ARRAYS $1 > $2 
}

compile_with_dcir_cc_opt(){
  $mlir_opt --affine-loop-invariant-code-motion $1 | \
  $mlir_opt --affine-scalrep | \
  $mlir_opt --lower-affine | \
  $mlir_opt --cse --inline > $2
}

compile_with_dcir_conversion(){
  $sdfg_opt --convert-to-sdfg $1 | \
  $sdfg_translate --mlir-to-sdfg > $2
}

compile_with_dcir(){
  src=$1
  src_name=$(basename ${src%.*})
  mkdir tmp_folder

  save_time "compile_with_dcir_cgeist $src tmp_folder/cgeist.mlir" \
    $out_dir/$src_name/dcir/cgeist.txt
  save_time "compile_with_dcir_cc_opt tmp_folder/cgeist.mlir tmp_folder/opt.mlir" \
    $out_dir/$src_name/dcir/cc_opt.txt
  save_time "compile_with_dcir_conversion tmp_folder/opt.mlir tmp_folder/translated.sdfg" \
    $out_dir/$src_name/dcir/conversion.txt

  $python opt_compile.py tmp_folder/translated.sdfg \
  $out_dir/$src_name/dcir/dc_opt.txt $out_dir/$src_name/dcir/dace_compile.txt

  rm -r tmp_folder
}

benchmarks=$(find benchmarks/* -name '*.c' -not -path "benchmarks/utilities/*")
for filename in $benchmarks; do
  src=$filename
  src_name=$(basename ${src%.*})

  mkdir -p $out_dir/$src_name

  ### GCC ###
  mkdir -p $out_dir/$src_name/gcc
  save_time "compile_with_gcc $src" $out_dir/$src_name/gcc/total.txt
  
  ### Clang ###
  mkdir -p $out_dir/$src_name/clang
  save_time "compile_with_clang $src" $out_dir/$src_name/clang/total.txt

  ### MLIR ###
  mkdir -p $out_dir/$src_name/mlir
  save_time "compile_with_mlir $src" $out_dir/$src_name/mlir/total.txt

  ### DCIR ###
  mkdir -p $out_dir/$src_name/dcir
  save_time "compile_with_dcir $src" $out_dir/$src_name/dcir/total.txt

done
