module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  func.func @main() -> i8 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i8 = arith.constant 5 : i8
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.call @malloc(%c1_i64) : (i64) -> !llvm.ptr<i8>
    %1 = llvm.call @malloc(%c1_i64) : (i64) -> !llvm.ptr<i8>
    affine.for %arg0 = 0 to 10000000 {
      %3 = arith.index_cast %arg0 : index to i8
      llvm.store %c5_i8, %0 : !llvm.ptr<i8>
      llvm.store %3, %1 : !llvm.ptr<i8>
    }
    %2 = llvm.load %0 : !llvm.ptr<i8>
    return %2 : i8
  }
}
