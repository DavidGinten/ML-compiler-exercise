module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_5xf32(dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<5 x f32>
  llvm.mlir.global private constant @__constant_5x4xf32(dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<5 x array<4 x f32>>
  llvm.mlir.global private constant @__constant_3x4xf32(dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x array<4 x f32>>
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.addressof @__constant_3x4xf32 : !llvm.ptr
    %5 = llvm.mlir.addressof @__constant_5x4xf32 : !llvm.ptr
    %6 = llvm.mlir.addressof @__constant_5xf32 : !llvm.ptr
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.constant(3 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(4 : index) : i64
    %12 = llvm.mlir.constant(5 : index) : i64
    %13 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x array<4 x f32>>
    %14 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x array<4 x f32>>
    %15 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x f32>
    %16 = llvm.getelementptr %3[12] : (!llvm.ptr) -> !llvm.ptr, f32
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.add %17, %7 : i64
    %19 = llvm.call @malloc(%18) : (i64) -> !llvm.ptr
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.sub %7, %10 : i64
    %22 = llvm.add %20, %21 : i64
    %23 = llvm.urem %22, %7 : i64
    %24 = llvm.sub %22, %23 : i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
    llvm.br ^bb1(%8 : i64)
  ^bb1(%26: i64):  // 2 preds: ^bb0, ^bb5
    %27 = llvm.icmp "slt" %26, %9 : i64
    llvm.cond_br %27, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%8 : i64)
  ^bb3(%28: i64):  // 2 preds: ^bb2, ^bb4
    %29 = llvm.icmp "slt" %28, %11 : i64
    llvm.cond_br %29, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %30 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %31 = llvm.mul %26, %arg5 overflow<nsw, nuw> : i64
    %32 = llvm.mul %28, %arg6 overflow<nsw, nuw> : i64
    %33 = llvm.add %31, %32 overflow<nsw, nuw> : i64
    %34 = llvm.getelementptr inbounds|nuw %30[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %35 = llvm.load %34 : !llvm.ptr -> f32
    %36 = llvm.mul %26, %11 overflow<nsw, nuw> : i64
    %37 = llvm.add %36, %28 overflow<nsw, nuw> : i64
    %38 = llvm.getelementptr inbounds|nuw %13[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %39 = llvm.load %38 : !llvm.ptr -> f32
    %40 = llvm.fadd %35, %39 : f32
    %41 = llvm.getelementptr inbounds|nuw %25[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %40, %41 : f32, !llvm.ptr
    %42 = llvm.add %28, %10 : i64
    llvm.br ^bb3(%42 : i64)
  ^bb5:  // pred: ^bb3
    %43 = llvm.add %26, %10 : i64
    llvm.br ^bb1(%43 : i64)
  ^bb6:  // pred: ^bb1
    %44 = llvm.getelementptr %3[20] : (!llvm.ptr) -> !llvm.ptr, f32
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.add %45, %7 : i64
    %47 = llvm.call @malloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.add %48, %21 : i64
    %50 = llvm.urem %49, %7 : i64
    %51 = llvm.sub %49, %50 : i64
    %52 = llvm.inttoptr %51 : i64 to !llvm.ptr
    llvm.br ^bb7(%8 : i64)
  ^bb7(%53: i64):  // 2 preds: ^bb6, ^bb11
    %54 = llvm.icmp "slt" %53, %11 : i64
    llvm.cond_br %54, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%8 : i64)
  ^bb9(%55: i64):  // 2 preds: ^bb8, ^bb10
    %56 = llvm.icmp "slt" %55, %12 : i64
    llvm.cond_br %56, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %57 = llvm.mul %55, %11 overflow<nsw, nuw> : i64
    %58 = llvm.add %57, %53 overflow<nsw, nuw> : i64
    %59 = llvm.getelementptr inbounds|nuw %14[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.load %59 : !llvm.ptr -> f32
    %61 = llvm.mul %53, %12 overflow<nsw, nuw> : i64
    %62 = llvm.add %61, %55 overflow<nsw, nuw> : i64
    %63 = llvm.getelementptr inbounds|nuw %52[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %60, %63 : f32, !llvm.ptr
    %64 = llvm.add %55, %10 : i64
    llvm.br ^bb9(%64 : i64)
  ^bb11:  // pred: ^bb9
    %65 = llvm.add %53, %10 : i64
    llvm.br ^bb7(%65 : i64)
  ^bb12:  // pred: ^bb7
    %66 = llvm.getelementptr %3[15] : (!llvm.ptr) -> !llvm.ptr, f32
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.add %67, %7 : i64
    %69 = llvm.call @malloc(%68) : (i64) -> !llvm.ptr
    %70 = llvm.ptrtoint %69 : !llvm.ptr to i64
    %71 = llvm.add %70, %21 : i64
    %72 = llvm.urem %71, %7 : i64
    %73 = llvm.sub %71, %72 : i64
    %74 = llvm.inttoptr %73 : i64 to !llvm.ptr
    %75 = llvm.insertvalue %69, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.insertvalue %74, %75[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %8, %76[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.insertvalue %9, %77[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.insertvalue %12, %78[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.insertvalue %12, %79[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.insertvalue %10, %80[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb13(%8 : i64)
  ^bb13(%82: i64):  // 2 preds: ^bb12, ^bb17
    %83 = llvm.icmp "slt" %82, %9 : i64
    llvm.cond_br %83, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%8 : i64)
  ^bb15(%84: i64):  // 2 preds: ^bb14, ^bb16
    %85 = llvm.icmp "slt" %84, %12 : i64
    llvm.cond_br %85, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %86 = llvm.mul %82, %12 overflow<nsw, nuw> : i64
    %87 = llvm.add %86, %84 overflow<nsw, nuw> : i64
    %88 = llvm.getelementptr inbounds|nuw %74[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %88 : f32, !llvm.ptr
    %89 = llvm.add %84, %10 : i64
    llvm.br ^bb15(%89 : i64)
  ^bb17:  // pred: ^bb15
    %90 = llvm.add %82, %10 : i64
    llvm.br ^bb13(%90 : i64)
  ^bb18:  // pred: ^bb13
    llvm.br ^bb19(%8 : i64)
  ^bb19(%91: i64):  // 2 preds: ^bb18, ^bb26
    %92 = llvm.icmp "slt" %91, %9 : i64
    llvm.cond_br %92, ^bb20, ^bb27
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%8 : i64)
  ^bb21(%93: i64):  // 2 preds: ^bb20, ^bb25
    %94 = llvm.icmp "slt" %93, %12 : i64
    llvm.cond_br %94, ^bb22, ^bb26
  ^bb22:  // pred: ^bb21
    llvm.br ^bb23(%8 : i64)
  ^bb23(%95: i64):  // 2 preds: ^bb22, ^bb24
    %96 = llvm.icmp "slt" %95, %11 : i64
    llvm.cond_br %96, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    %97 = llvm.mul %91, %11 overflow<nsw, nuw> : i64
    %98 = llvm.add %97, %95 overflow<nsw, nuw> : i64
    %99 = llvm.getelementptr inbounds|nuw %25[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %100 = llvm.load %99 : !llvm.ptr -> f32
    %101 = llvm.mul %95, %12 overflow<nsw, nuw> : i64
    %102 = llvm.add %101, %93 overflow<nsw, nuw> : i64
    %103 = llvm.getelementptr inbounds|nuw %52[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %104 = llvm.load %103 : !llvm.ptr -> f32
    %105 = llvm.mul %91, %12 overflow<nsw, nuw> : i64
    %106 = llvm.add %105, %93 overflow<nsw, nuw> : i64
    %107 = llvm.getelementptr inbounds|nuw %74[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.load %107 : !llvm.ptr -> f32
    %109 = llvm.fmul %100, %104 : f32
    %110 = llvm.fadd %108, %109 : f32
    llvm.store %110, %107 : f32, !llvm.ptr
    %111 = llvm.add %95, %10 : i64
    llvm.br ^bb23(%111 : i64)
  ^bb25:  // pred: ^bb23
    %112 = llvm.add %93, %10 : i64
    llvm.br ^bb21(%112 : i64)
  ^bb26:  // pred: ^bb21
    %113 = llvm.add %91, %10 : i64
    llvm.br ^bb19(%113 : i64)
  ^bb27:  // pred: ^bb19
    llvm.br ^bb28(%8 : i64)
  ^bb28(%114: i64):  // 2 preds: ^bb27, ^bb32
    %115 = llvm.icmp "slt" %114, %9 : i64
    llvm.cond_br %115, ^bb29, ^bb33
  ^bb29:  // pred: ^bb28
    llvm.br ^bb30(%8 : i64)
  ^bb30(%116: i64):  // 2 preds: ^bb29, ^bb31
    %117 = llvm.icmp "slt" %116, %12 : i64
    llvm.cond_br %117, ^bb31, ^bb32
  ^bb31:  // pred: ^bb30
    %118 = llvm.mul %114, %12 overflow<nsw, nuw> : i64
    %119 = llvm.add %118, %116 overflow<nsw, nuw> : i64
    %120 = llvm.getelementptr inbounds|nuw %74[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %121 = llvm.load %120 : !llvm.ptr -> f32
    %122 = llvm.getelementptr inbounds|nuw %15[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %123 = llvm.load %122 : !llvm.ptr -> f32
    %124 = llvm.fadd %121, %123 : f32
    llvm.store %124, %120 : f32, !llvm.ptr
    %125 = llvm.add %116, %10 : i64
    llvm.br ^bb30(%125 : i64)
  ^bb32:  // pred: ^bb30
    %126 = llvm.add %114, %10 : i64
    llvm.br ^bb28(%126 : i64)
  ^bb33:  // pred: ^bb28
    llvm.br ^bb34(%8 : i64)
  ^bb34(%127: i64):  // 2 preds: ^bb33, ^bb38
    %128 = llvm.icmp "slt" %127, %9 : i64
    llvm.cond_br %128, ^bb35, ^bb39
  ^bb35:  // pred: ^bb34
    llvm.br ^bb36(%8 : i64)
  ^bb36(%129: i64):  // 2 preds: ^bb35, ^bb37
    %130 = llvm.icmp "slt" %129, %12 : i64
    llvm.cond_br %130, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    %131 = llvm.mul %127, %12 overflow<nsw, nuw> : i64
    %132 = llvm.add %131, %129 overflow<nsw, nuw> : i64
    %133 = llvm.getelementptr inbounds|nuw %74[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.load %133 : !llvm.ptr -> f32
    %135 = llvm.fcmp "ult" %134, %2 : f32
    %136 = llvm.select %135, %2, %134 : i1, f32
    %137 = llvm.fcmp "ugt" %136, %1 : f32
    %138 = llvm.select %137, %1, %136 : i1, f32
    llvm.store %138, %133 : f32, !llvm.ptr
    %139 = llvm.add %129, %10 : i64
    llvm.br ^bb36(%139 : i64)
  ^bb38:  // pred: ^bb36
    %140 = llvm.add %127, %10 : i64
    llvm.br ^bb34(%140 : i64)
  ^bb39:  // pred: ^bb34
    llvm.call @free(%19) : (!llvm.ptr) -> ()
    llvm.call @free(%47) : (!llvm.ptr) -> ()
    llvm.return %81 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}