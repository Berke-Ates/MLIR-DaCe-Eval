module {
  sdfg.sdfg {entry = @init_2} (%arg0: !sdfg.array<sym("s_0")xi32>, %arg1: !sdfg.array<sym("s_1")xi32>) -> (%arg2: !sdfg.array<i32>){
    %0 = sdfg.alloc {name = "_addi_tmp_10", transient} () : !sdfg.array<i32>
    %1 = sdfg.alloc {name = "_load_tmp_7", transient} () : !sdfg.array<i32>
    %2 = sdfg.alloc {name = "_load_tmp_5", transient} () : !sdfg.array<i32>
    %3 = sdfg.alloc {name = "_constant_tmp_4", transient} () : !sdfg.array<index>
    sdfg.state @init_2{
    }
    sdfg.state @constant_3{
      %4 = sdfg.tasklet () -> (index){
        %c0 = arith.constant 0 : index
        sdfg.return %c0 : index
      }
      sdfg.store %4, %3[] : index -> !sdfg.array<index>
      %5 = sdfg.load %3[] : !sdfg.array<index> -> index
    }
    sdfg.state @load_6{
      %4 = sdfg.load %3[] : !sdfg.array<index> -> index
      %5 = sdfg.load %arg0[%4] : !sdfg.array<sym("s_0")xi32> -> i32
      sdfg.store %5, %2[] : i32 -> !sdfg.array<i32>
      %6 = sdfg.load %2[] : !sdfg.array<i32> -> i32
    }
    sdfg.state @load_8{
      %4 = sdfg.load %3[] : !sdfg.array<index> -> index
      %5 = sdfg.load %arg1[%4] : !sdfg.array<sym("s_1")xi32> -> i32
      sdfg.store %5, %1[] : i32 -> !sdfg.array<i32>
      %6 = sdfg.load %1[] : !sdfg.array<i32> -> i32
    }
    sdfg.state @addi_9{
      %4 = sdfg.load %2[] : !sdfg.array<i32> -> i32
      %5 = sdfg.load %1[] : !sdfg.array<i32> -> i32
      %6 = sdfg.tasklet (%4 as %arg3: i32, %5 as %arg4: i32) -> (i32){
        %8 = arith.addi %arg3, %arg4 : i32
        sdfg.return %8 : i32
      }
      sdfg.store %6, %0[] : i32 -> !sdfg.array<i32>
      %7 = sdfg.load %0[] : !sdfg.array<i32> -> i32
    }
    sdfg.state @return_11{
      %4 = sdfg.load %0[] : !sdfg.array<i32> -> i32
      sdfg.store %4, %arg2[] : i32 -> !sdfg.array<i32>
    }
    sdfg.edge {assign = [], condition = "1"} @init_2 -> @constant_3
    sdfg.edge {assign = [], condition = "1"} @constant_3 -> @load_6
    sdfg.edge {assign = [], condition = "1"} @load_6 -> @load_8
    sdfg.edge {assign = [], condition = "1"} @load_8 -> @addi_9
    sdfg.edge {assign = [], condition = "1"} @addi_9 -> @return_11
  }
}

