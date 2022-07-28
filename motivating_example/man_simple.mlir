module {
  func.func @main() -> i8 {
    %c5_i8 = arith.constant 5 : i8
    // %c1_i64 = arith.constant 1 : index
    
    // %0 = llvm.call @malloc(%c1_i64) : (i64) -> !llvm.ptr<i8>
    // %1 = llvm.call @malloc(%c1_i64) : (i64) -> !llvm.ptr<i8>
    %0 = memref.alloc() : memref<i8>
    %1 = memref.alloc() : memref<i8>

    affine.for %arg0 = 0 to 10000000 {
      %3 = arith.index_cast %arg0 : index to i8

      // llvm.store %c5_i8, %0 : !llvm.ptr<i8>
      // llvm.store %3, %1 : !llvm.ptr<i8>
      memref.store %c5_i8, %0[] : memref<i8>
      memref.store %3, %1[] : memref<i8>
    }

    // %2 = llvm.load %0 : !llvm.ptr<i8>
    %2 = memref.load %0[] : memref<i8>

    return %2 : i8
  }
}

