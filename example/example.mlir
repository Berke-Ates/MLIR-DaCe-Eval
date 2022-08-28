module {
  func.func @fName(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<?xi32>
    %1 = memref.load %arg1[%c0] : memref<?xi32>
    %2 = arith.addi %0, %1 : i32
    return %2 : i32
  }
}

