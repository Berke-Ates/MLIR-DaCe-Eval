#!/bin/bash

# Usage: ./plot.sh <txt timings file> <repetitions>

# Settings
labels=("GCC" "Clang" "Dace" "Polygeist + MLIR" "MLIR-DaCe")
repetitions=$2
timings_dir=./timings
out_dir=./plots

# Create output directory
if [ ! -d $out_dir ]; then
  mkdir -p $out_dir;
fi

src=$1
src_name=$(basename ${src%.*})

timings=$timings_dir/$src_name.csv
rm $timings
touch $timings

for i in ${!labels[@]}; do
  if [ $i -ne 0 ]; then
    echo -ne "," >> $timings
  fi

  echo -ne "${labels[$i]}" >> $timings
done

echo -ne "\n" >> $timings

for i in $(seq 1 $repetitions); do

  for j in ${!labels[@]}; do
    if [ $j -ne 0 ]; then
      echo -ne "," >> $timings
    fi
    
    num=$((($repetitions + 2) * $j + $i + 1))
    line=$(sed "${num}q;d" $src)
    echo -ne "$line" >> $timings
  done

  echo -ne "\n" >> $timings
done
