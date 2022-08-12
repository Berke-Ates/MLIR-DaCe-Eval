#!/bin/bash

# Usage: ./run_all.sh

# Settings
out_dir=./timings
result_dir=./out

# Create output directory
if [ ! -d $out_dir ]; then
  mkdir -p $out_dir;
fi
rm -rf $out_dir/* # Clear output directory

benchmarks=$(find benchmarks/* -name '*.c' -not -path "benchmarks/utilities/*")
total=$(echo "$benchmarks" | wc -l)
count=0

for filename in $benchmarks; do
    bname="$(basename $filename .c)"
    count=$((count+1))
    diff=$(($total - $count))
    percent=$(($count * 100 / $total))

    prog=''
    for i in $(seq 1 $count); do
      prog="$prog#"
    done

    for i in $(seq 1 $diff); do
      prog="$prog-"
    done

    echo -ne "\033[2K\r"
    echo -ne "$prog ($percent%) ($bname)"

    ./run.sh $filename &> /dev/null
    cp $result_dir/timings.txt $out_dir/$bname.txt
done

echo ""