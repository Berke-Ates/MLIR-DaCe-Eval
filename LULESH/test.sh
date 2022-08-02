#!/bin/bash

# Usage: ./test.sh

# This scripts runs all external functions to check that all of them function
# properly. WARNING: This will take a very long time

fmt_start="\n🔥 %s %s\n"
fmt_succ="    ✅ %s\n"
fmt_err="    ❌ %s\n"

for file in ext/*; do 
  src_name=$(basename ${file%.*})
  printf "$fmt_start" "Running:" $src_name
  
  ./run.sh $file > /dev/null

  if [ $? -eq 0 ]; then
    printf "$fmt_succ" "Output correct"
  else
    printf "$fmt_err" "Output wrong!"
  fi
done
