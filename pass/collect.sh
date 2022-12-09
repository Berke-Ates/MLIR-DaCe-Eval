#!/bin/bash

# Check args
if [ $# -ne 2 ]; then
  echo "Usage: ./collect.sh <Input Dir> <Output File>"
  exit 1
fi

# Read args
input_dir=$1
output_file=$2

benchmarks=$(find "$input_dir"/* -name '*.txt')

subgraph_fusions=0
loop_to_map=0
refine_nested_access=0
map_collapse=0
consolidated_edges=0
eliminated_arrays=0
fused_states=0
inferred_optional_arrays=0
inlined_sdfgs=0
promoted_scalars_to_symbols=0
propagated_constants=0
removed_unused_symbols=0
statically_allocated_transient_arrays=0
moved_loops_into_maps=0
state_fusions=0

collect_pass() {
  pattern=$1
  file=$2
  col=$3
  local -n var_ref=$4

  if ! grep -Pq "$pattern" "$file"; then
    return 0
  fi

  num=$(grep -P "$pattern" "$file" | awk "{print \$$col}" | paste -sd+ | bc)
  var_ref=$((var_ref + num))

  sed -e "s/$pattern//g" -i "$file"
  sed -e "/^[[:space:]]*$/d" -i "$file"
}

for benchmark in $benchmarks; do
  sed -e "s/SDFG .: //g" -i "$benchmark"
  sort -o "$benchmark" "$benchmark"
  sed -e "s/:.*//g" -i "$benchmark"
  sed -e "s/\.//g" -i "$benchmark"

  collect_pass "Applied .* SubgraphFusion" "$benchmark" 2 subgraph_fusions
  collect_pass "Applied .* LoopToMap" "$benchmark" 2 loop_to_map
  collect_pass ", .* RefineNestedAccess" "$benchmark" 2 refine_nested_access
  collect_pass "Applied .* MapCollapse" "$benchmark" 2 map_collapse
  collect_pass "Consolidated .* edges" "$benchmark" 2 consolidated_edges
  collect_pass "Eliminated .* arrays" "$benchmark" 2 eliminated_arrays
  collect_pass "Fused .* states" "$benchmark" 2 fused_states
  collect_pass "Inferred .* optional arrays" "$benchmark" 2 inferred_optional_arrays
  collect_pass "Inlined .* SDFGs" "$benchmark" 2 inlined_sdfgs
  collect_pass "Promoted .* scalars to symbols" "$benchmark" 2 promoted_scalars_to_symbols
  collect_pass "Propagated .* constants" "$benchmark" 2 propagated_constants
  collect_pass "Removed .* unused symbols" "$benchmark" 2 removed_unused_symbols
  collect_pass "Statically allocating .* transient arrays" "$benchmark" 3 statically_allocated_transient_arrays
  collect_pass "Applied .* MoveLoopIntoMap" "$benchmark" 2 moved_loops_into_maps
  collect_pass "Applied .* StateFusion" "$benchmark" 2 state_fusions
done

{
  echo "Applied SubgraphFusion: $subgraph_fusions"
  echo "Applied LoopToMap: $loop_to_map"
  echo "RefineNestedAccess: $refine_nested_access"
  echo "Applied MapCollapse: $map_collapse"
  echo "Consolidated edges: $consolidated_edges"
  echo "Eliminated Arrays: $eliminated_arrays"
  echo "Fused states: $fused_states"
  echo "Inferred optional arrays: $inferred_optional_arrays"
  echo "Inlined SDFGs: $inlined_sdfgs"
  echo "Promoted scalars to symbols: $promoted_scalars_to_symbols"
  echo "Propagated constants: $propagated_constants"
  echo "Removed unused symbols: $removed_unused_symbols"
  echo "Statically allocating transient arrays: $statically_allocated_transient_arrays"
  echo "Applied MoveLoopIntoMap: $moved_loops_into_maps"
  echo "Applied StateFusion: $state_fusions"
} >>"$output_file"
