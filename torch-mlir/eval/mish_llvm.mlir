module attributes {llvm.data_layout = "", torch.debug_module_name = "Mish"} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private @global_seed(0 : i64) {addr_space = 0 : i32} : i64
  llvm.func @refbackend_consume_func_return_mrf32(%arg0: i64, %arg1: !llvm.ptr<i8>) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)> 
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.store %2, %4 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.call @_mlir_ciface_refbackend_consume_func_return_mrf32(%4) : (!llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_refbackend_consume_func_return_mrf32(!llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.mlir.global private constant @__constant_8x32x224x224xf32(dense<1.000000e+00> : tensor<8x32x224x224xf32>) {addr_space = 0 : i32} : !llvm.array<8 x array<32 x array<224 x array<224 x f32>>>>
  llvm.func @forward(%arg0: i64, %arg1: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(8 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(32 : index) : i64
    %7 = llvm.mlir.constant(224 : index) : i64
    %8 = llvm.extractvalue %2[1] : !llvm.struct<(i64, ptr<i8>)> 
    %9 = llvm.bitcast %8 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %10 = llvm.load %9 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %11 = llvm.mlir.constant(8 : index) : i64
    %12 = llvm.mlir.constant(32 : index) : i64
    %13 = llvm.mlir.constant(224 : index) : i64
    %14 = llvm.mlir.constant(224 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(50176 : index) : i64
    %17 = llvm.mlir.constant(1605632 : index) : i64
    %18 = llvm.mlir.constant(12845056 : index) : i64
    %19 = llvm.mlir.null : !llvm.ptr<f32>
    %20 = llvm.getelementptr %19[12845056] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %21 = llvm.ptrtoint %20 : !llvm.ptr<f32> to i64
    %22 = llvm.mlir.addressof @__constant_8x32x224x224xf32 : !llvm.ptr<array<8 x array<32 x array<224 x array<224 x f32>>>>>
    %23 = llvm.getelementptr %22[0, 0, 0, 0, 0] : (!llvm.ptr<array<8 x array<32 x array<224 x array<224 x f32>>>>>) -> !llvm.ptr<f32>
    %24 = llvm.mlir.constant(3735928559 : index) : i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr<f32>
    %26 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %23, %27[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.insertvalue %11, %30[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.insertvalue %12, %31[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.insertvalue %13, %32[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.insertvalue %14, %33[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.insertvalue %17, %34[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.insertvalue %16, %35[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %38 = llvm.insertvalue %15, %37[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %39 = llvm.mlir.constant(8 : index) : i64
    %40 = llvm.mlir.constant(32 : index) : i64
    %41 = llvm.mlir.constant(224 : index) : i64
    %42 = llvm.mlir.constant(224 : index) : i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.mlir.constant(50176 : index) : i64
    %45 = llvm.mlir.constant(1605632 : index) : i64
    %46 = llvm.mlir.constant(12845056 : index) : i64
    %47 = llvm.mlir.null : !llvm.ptr<f32>
    %48 = llvm.getelementptr %47[12845056] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %49 = llvm.ptrtoint %48 : !llvm.ptr<f32> to i64
    %50 = llvm.mlir.constant(128 : index) : i64
    %51 = llvm.add %49, %50  : i64
    %52 = llvm.call @malloc(%51) : (i64) -> !llvm.ptr<i8>
    %53 = llvm.bitcast %52 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %54 = llvm.ptrtoint %53 : !llvm.ptr<f32> to i64
    %55 = llvm.mlir.constant(1 : index) : i64
    %56 = llvm.sub %50, %55  : i64
    %57 = llvm.add %54, %56  : i64
    %58 = llvm.urem %57, %50  : i64
    %59 = llvm.sub %57, %58  : i64
    %60 = llvm.inttoptr %59 : i64 to !llvm.ptr<f32>
    %61 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %62 = llvm.insertvalue %53, %61[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %63 = llvm.insertvalue %60, %62[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %64 = llvm.mlir.constant(0 : index) : i64
    %65 = llvm.insertvalue %64, %63[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %66 = llvm.insertvalue %39, %65[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %67 = llvm.insertvalue %40, %66[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.insertvalue %41, %67[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = llvm.insertvalue %42, %68[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %70 = llvm.insertvalue %45, %69[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.insertvalue %44, %70[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %72 = llvm.insertvalue %42, %71[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %73 = llvm.insertvalue %43, %72[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb1(%3 : i64)
  ^bb1(%74: i64):  // 2 preds: ^bb0, ^bb11
    %75 = llvm.icmp "slt" %74, %4 : i64
    llvm.cond_br %75, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%76: i64):  // 2 preds: ^bb2, ^bb10
    %77 = llvm.icmp "slt" %76, %6 : i64
    llvm.cond_br %77, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%3 : i64)
  ^bb5(%78: i64):  // 2 preds: ^bb4, ^bb9
    %79 = llvm.icmp "slt" %78, %7 : i64
    llvm.cond_br %79, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%3 : i64)
  ^bb7(%80: i64):  // 2 preds: ^bb6, ^bb8
    %81 = llvm.icmp "slt" %80, %7 : i64
    llvm.cond_br %81, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %82 = llvm.extractvalue %10[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %83 = llvm.mlir.constant(1605632 : index) : i64
    %84 = llvm.mul %74, %83  : i64
    %85 = llvm.mlir.constant(50176 : index) : i64
    %86 = llvm.mul %76, %85  : i64
    %87 = llvm.add %84, %86  : i64
    %88 = llvm.mlir.constant(224 : index) : i64
    %89 = llvm.mul %78, %88  : i64
    %90 = llvm.add %87, %89  : i64
    %91 = llvm.add %90, %80  : i64
    %92 = llvm.getelementptr %82[%91] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %93 = llvm.load %92 : !llvm.ptr<f32>
    %94 = "llvm.intr.exp"(%93) : (f32) -> f32
    %95 = llvm.mlir.constant(1605632 : index) : i64
    %96 = llvm.mul %74, %95  : i64
    %97 = llvm.mlir.constant(50176 : index) : i64
    %98 = llvm.mul %76, %97  : i64
    %99 = llvm.add %96, %98  : i64
    %100 = llvm.mlir.constant(224 : index) : i64
    %101 = llvm.mul %78, %100  : i64
    %102 = llvm.add %99, %101  : i64
    %103 = llvm.add %102, %80  : i64
    %104 = llvm.getelementptr %60[%103] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %94, %104 : !llvm.ptr<f32>
    %105 = llvm.add %80, %5  : i64
    llvm.br ^bb7(%105 : i64)
  ^bb9:  // pred: ^bb7
    %106 = llvm.add %78, %5  : i64
    llvm.br ^bb5(%106 : i64)
  ^bb10:  // pred: ^bb5
    %107 = llvm.add %76, %5  : i64
    llvm.br ^bb3(%107 : i64)
  ^bb11:  // pred: ^bb3
    %108 = llvm.add %74, %5  : i64
    llvm.br ^bb1(%108 : i64)
  ^bb12:  // pred: ^bb1
    %109 = llvm.mlir.constant(8 : index) : i64
    %110 = llvm.mlir.constant(32 : index) : i64
    %111 = llvm.mlir.constant(224 : index) : i64
    %112 = llvm.mlir.constant(224 : index) : i64
    %113 = llvm.mlir.constant(1 : index) : i64
    %114 = llvm.mlir.constant(50176 : index) : i64
    %115 = llvm.mlir.constant(1605632 : index) : i64
    %116 = llvm.mlir.constant(12845056 : index) : i64
    %117 = llvm.mlir.null : !llvm.ptr<f32>
    %118 = llvm.getelementptr %117[12845056] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %119 = llvm.ptrtoint %118 : !llvm.ptr<f32> to i64
    %120 = llvm.mlir.constant(128 : index) : i64
    %121 = llvm.add %119, %120  : i64
    %122 = llvm.call @malloc(%121) : (i64) -> !llvm.ptr<i8>
    %123 = llvm.bitcast %122 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %124 = llvm.ptrtoint %123 : !llvm.ptr<f32> to i64
    %125 = llvm.mlir.constant(1 : index) : i64
    %126 = llvm.sub %120, %125  : i64
    %127 = llvm.add %124, %126  : i64
    %128 = llvm.urem %127, %120  : i64
    %129 = llvm.sub %127, %128  : i64
    %130 = llvm.inttoptr %129 : i64 to !llvm.ptr<f32>
    %131 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %132 = llvm.insertvalue %123, %131[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %133 = llvm.insertvalue %130, %132[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %134 = llvm.mlir.constant(0 : index) : i64
    %135 = llvm.insertvalue %134, %133[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %136 = llvm.insertvalue %109, %135[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %137 = llvm.insertvalue %110, %136[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %138 = llvm.insertvalue %111, %137[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %139 = llvm.insertvalue %112, %138[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %140 = llvm.insertvalue %115, %139[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %141 = llvm.insertvalue %114, %140[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %142 = llvm.insertvalue %112, %141[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %143 = llvm.insertvalue %113, %142[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb13(%3 : i64)
  ^bb13(%144: i64):  // 2 preds: ^bb12, ^bb23
    %145 = llvm.icmp "slt" %144, %4 : i64
    llvm.cond_br %145, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%3 : i64)
  ^bb15(%146: i64):  // 2 preds: ^bb14, ^bb22
    %147 = llvm.icmp "slt" %146, %6 : i64
    llvm.cond_br %147, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%3 : i64)
  ^bb17(%148: i64):  // 2 preds: ^bb16, ^bb21
    %149 = llvm.icmp "slt" %148, %7 : i64
    llvm.cond_br %149, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%3 : i64)
  ^bb19(%150: i64):  // 2 preds: ^bb18, ^bb20
    %151 = llvm.icmp "slt" %150, %7 : i64
    llvm.cond_br %151, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %152 = llvm.mlir.constant(1605632 : index) : i64
    %153 = llvm.mul %144, %152  : i64
    %154 = llvm.mlir.constant(50176 : index) : i64
    %155 = llvm.mul %146, %154  : i64
    %156 = llvm.add %153, %155  : i64
    %157 = llvm.mlir.constant(224 : index) : i64
    %158 = llvm.mul %148, %157  : i64
    %159 = llvm.add %156, %158  : i64
    %160 = llvm.add %159, %150  : i64
    %161 = llvm.getelementptr %60[%160] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %162 = llvm.load %161 : !llvm.ptr<f32>
    %163 = llvm.mlir.constant(1605632 : index) : i64
    %164 = llvm.mul %144, %163  : i64
    %165 = llvm.mlir.constant(50176 : index) : i64
    %166 = llvm.mul %146, %165  : i64
    %167 = llvm.add %164, %166  : i64
    %168 = llvm.mlir.constant(224 : index) : i64
    %169 = llvm.mul %148, %168  : i64
    %170 = llvm.add %167, %169  : i64
    %171 = llvm.add %170, %150  : i64
    %172 = llvm.getelementptr %23[%171] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %173 = llvm.load %172 : !llvm.ptr<f32>
    %174 = llvm.fadd %162, %173  : f32
    %175 = llvm.mlir.constant(1605632 : index) : i64
    %176 = llvm.mul %144, %175  : i64
    %177 = llvm.mlir.constant(50176 : index) : i64
    %178 = llvm.mul %146, %177  : i64
    %179 = llvm.add %176, %178  : i64
    %180 = llvm.mlir.constant(224 : index) : i64
    %181 = llvm.mul %148, %180  : i64
    %182 = llvm.add %179, %181  : i64
    %183 = llvm.add %182, %150  : i64
    %184 = llvm.getelementptr %130[%183] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %174, %184 : !llvm.ptr<f32>
    %185 = llvm.add %150, %5  : i64
    llvm.br ^bb19(%185 : i64)
  ^bb21:  // pred: ^bb19
    %186 = llvm.add %148, %5  : i64
    llvm.br ^bb17(%186 : i64)
  ^bb22:  // pred: ^bb17
    %187 = llvm.add %146, %5  : i64
    llvm.br ^bb15(%187 : i64)
  ^bb23:  // pred: ^bb15
    %188 = llvm.add %144, %5  : i64
    llvm.br ^bb13(%188 : i64)
  ^bb24:  // pred: ^bb13
    %189 = llvm.mlir.constant(8 : index) : i64
    %190 = llvm.mlir.constant(32 : index) : i64
    %191 = llvm.mlir.constant(224 : index) : i64
    %192 = llvm.mlir.constant(224 : index) : i64
    %193 = llvm.mlir.constant(1 : index) : i64
    %194 = llvm.mlir.constant(50176 : index) : i64
    %195 = llvm.mlir.constant(1605632 : index) : i64
    %196 = llvm.mlir.constant(12845056 : index) : i64
    %197 = llvm.mlir.null : !llvm.ptr<f32>
    %198 = llvm.getelementptr %197[12845056] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %199 = llvm.ptrtoint %198 : !llvm.ptr<f32> to i64
    %200 = llvm.mlir.constant(128 : index) : i64
    %201 = llvm.add %199, %200  : i64
    %202 = llvm.call @malloc(%201) : (i64) -> !llvm.ptr<i8>
    %203 = llvm.bitcast %202 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %204 = llvm.ptrtoint %203 : !llvm.ptr<f32> to i64
    %205 = llvm.mlir.constant(1 : index) : i64
    %206 = llvm.sub %200, %205  : i64
    %207 = llvm.add %204, %206  : i64
    %208 = llvm.urem %207, %200  : i64
    %209 = llvm.sub %207, %208  : i64
    %210 = llvm.inttoptr %209 : i64 to !llvm.ptr<f32>
    %211 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %212 = llvm.insertvalue %203, %211[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %213 = llvm.insertvalue %210, %212[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %214 = llvm.mlir.constant(0 : index) : i64
    %215 = llvm.insertvalue %214, %213[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %216 = llvm.insertvalue %189, %215[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %217 = llvm.insertvalue %190, %216[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %218 = llvm.insertvalue %191, %217[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %219 = llvm.insertvalue %192, %218[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %220 = llvm.insertvalue %195, %219[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %221 = llvm.insertvalue %194, %220[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %222 = llvm.insertvalue %192, %221[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %223 = llvm.insertvalue %193, %222[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb25(%3 : i64)
  ^bb25(%224: i64):  // 2 preds: ^bb24, ^bb35
    %225 = llvm.icmp "slt" %224, %4 : i64
    llvm.cond_br %225, ^bb26, ^bb36
  ^bb26:  // pred: ^bb25
    llvm.br ^bb27(%3 : i64)
  ^bb27(%226: i64):  // 2 preds: ^bb26, ^bb34
    %227 = llvm.icmp "slt" %226, %6 : i64
    llvm.cond_br %227, ^bb28, ^bb35
  ^bb28:  // pred: ^bb27
    llvm.br ^bb29(%3 : i64)
  ^bb29(%228: i64):  // 2 preds: ^bb28, ^bb33
    %229 = llvm.icmp "slt" %228, %7 : i64
    llvm.cond_br %229, ^bb30, ^bb34
  ^bb30:  // pred: ^bb29
    llvm.br ^bb31(%3 : i64)
  ^bb31(%230: i64):  // 2 preds: ^bb30, ^bb32
    %231 = llvm.icmp "slt" %230, %7 : i64
    llvm.cond_br %231, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %232 = llvm.mlir.constant(1605632 : index) : i64
    %233 = llvm.mul %224, %232  : i64
    %234 = llvm.mlir.constant(50176 : index) : i64
    %235 = llvm.mul %226, %234  : i64
    %236 = llvm.add %233, %235  : i64
    %237 = llvm.mlir.constant(224 : index) : i64
    %238 = llvm.mul %228, %237  : i64
    %239 = llvm.add %236, %238  : i64
    %240 = llvm.add %239, %230  : i64
    %241 = llvm.getelementptr %130[%240] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %242 = llvm.load %241 : !llvm.ptr<f32>
    %243 = "llvm.intr.log"(%242) : (f32) -> f32
    %244 = llvm.mlir.constant(1605632 : index) : i64
    %245 = llvm.mul %224, %244  : i64
    %246 = llvm.mlir.constant(50176 : index) : i64
    %247 = llvm.mul %226, %246  : i64
    %248 = llvm.add %245, %247  : i64
    %249 = llvm.mlir.constant(224 : index) : i64
    %250 = llvm.mul %228, %249  : i64
    %251 = llvm.add %248, %250  : i64
    %252 = llvm.add %251, %230  : i64
    %253 = llvm.getelementptr %210[%252] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %243, %253 : !llvm.ptr<f32>
    %254 = llvm.add %230, %5  : i64
    llvm.br ^bb31(%254 : i64)
  ^bb33:  // pred: ^bb31
    %255 = llvm.add %228, %5  : i64
    llvm.br ^bb29(%255 : i64)
  ^bb34:  // pred: ^bb29
    %256 = llvm.add %226, %5  : i64
    llvm.br ^bb27(%256 : i64)
  ^bb35:  // pred: ^bb27
    %257 = llvm.add %224, %5  : i64
    llvm.br ^bb25(%257 : i64)
  ^bb36:  // pred: ^bb25
    %258 = llvm.mlir.constant(1 : index) : i64
    %259 = llvm.alloca %258 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    llvm.store %223, %259 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %260 = llvm.bitcast %259 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>> to !llvm.ptr<i8>
    %261 = llvm.mlir.constant(4 : index) : i64
    %262 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %263 = llvm.insertvalue %261, %262[0] : !llvm.struct<(i64, ptr<i8>)> 
    %264 = llvm.insertvalue %260, %263[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @refbackend_consume_func_return_mrf32(%261, %260) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_forward(%arg0: !llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr<i8>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @forward(%1, %2) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
}
