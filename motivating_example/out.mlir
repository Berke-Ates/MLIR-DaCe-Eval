module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    %0 = memref.alloc() : memref<2147483647xi32>
    %1 = memref.alloc() : memref<2147483647xi32>
    affine.for %arg0 = 0 to 2147483647 {
      %5 = affine.load %0[%arg0] : memref<2147483647xi32>
      %6 = arith.addi %5, %c5_i32 : i32
      affine.store %6, %0[%arg0] : memref<2147483647xi32>
    }
    %2 = affine.load %0[0] : memref<2147483647xi32>
    %3 = arith.index_cast %2 : i32 to index
    affine.for %arg0 = 0 to %3 {
      affine.store %c5_i32, %1[0] : memref<2147483647xi32>
    }
    %4 = affine.load %1[0] : memref<2147483647xi32>
    return %4 : i32
  }
}
