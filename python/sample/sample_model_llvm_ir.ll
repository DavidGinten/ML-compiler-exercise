; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_5xf32 = private constant [5 x float] [float 0xBF990C2C00000000, float 0xBFB15F8600000000, float 0x3FAD55C400000000, float 0x3FDECF7440000000, float 0x3FD2B4C700000000], align 64
@__constant_5x4xf32 = private constant [5 x [4 x float]] [[4 x float] [float 0x3FD8353B00000000, float 0x3FA4D37000000000, float 0xBFD709DF40000000, float 0x3FDD86A8C0000000], [4 x float] [float 0x3FD7772C80000000, float 0xBF87B9D000000000, float 0xBFD2B50400000000, float 0xBFC8C9D600000000], [4 x float] [float 0xBFDC412A80000000, float 0x3FD534EF40000000, float 0xBFA6372E00000000, float 0x3FD60C2CC0000000], [4 x float] [float 0x3FC818EE00000000, float 0xBFB964A800000000, float 0xBFD4764D40000000, float 0x3FC6462E00000000], [4 x float] [float 0xBFD4885300000000, float 0xBFBC724900000000, float 0x3FD305FD80000000, float 0xBFD16CDBC0000000]], align 64
@__constant_3x4xf32 = private constant [3 x [4 x float]] [[4 x float] [float 0x3FCE43E000000000, float 0x3FCD01D180000000, float 0x3FE99DF180000000, float 0x3FC5A7F100000000], [4 x float] [float 0x3FD0F51400000000, float 0x3FE8B42820000000, float 0x3FC0687980000000, float 0x3FE7D8CB20000000], [4 x float] [float 0x3FE9BE32E0000000, float 0x3FE457BAA0000000, float 0x3FE2DE2340000000, float 0x3FE62F6DA0000000]], align 64

declare void @free(ptr)

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @sample_model(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = call ptr @malloc(i64 112)
  %9 = ptrtoint ptr %8 to i64
  %10 = add i64 %9, 63
  %11 = urem i64 %10, 64
  %12 = sub i64 %10, %11
  %13 = inttoptr i64 %12 to ptr
  br label %14

14:                                               ; preds = %35, %7
  %15 = phi i64 [ %36, %35 ], [ 0, %7 ]
  %16 = icmp slt i64 %15, 3
  br i1 %16, label %17, label %37

17:                                               ; preds = %14
  br label %18

18:                                               ; preds = %21, %17
  %19 = phi i64 [ %34, %21 ], [ 0, %17 ]
  %20 = icmp slt i64 %19, 4
  br i1 %20, label %21, label %35

21:                                               ; preds = %18
  %22 = getelementptr float, ptr %1, i64 %2
  %23 = mul nuw nsw i64 %15, %5
  %24 = mul nuw nsw i64 %19, %6
  %25 = add nuw nsw i64 %23, %24
  %26 = getelementptr inbounds nuw float, ptr %22, i64 %25
  %27 = load float, ptr %26, align 4
  %28 = mul nuw nsw i64 %15, 4
  %29 = add nuw nsw i64 %28, %19
  %30 = getelementptr inbounds nuw float, ptr @__constant_3x4xf32, i64 %29
  %31 = load float, ptr %30, align 4
  %32 = fadd float %27, %31
  %33 = getelementptr inbounds nuw float, ptr %13, i64 %29
  store float %32, ptr %33, align 4
  %34 = add i64 %19, 1
  br label %18

35:                                               ; preds = %18
  %36 = add i64 %15, 1
  br label %14

37:                                               ; preds = %14
  %38 = call ptr @malloc(i64 144)
  %39 = ptrtoint ptr %38 to i64
  %40 = add i64 %39, 63
  %41 = urem i64 %40, 64
  %42 = sub i64 %40, %41
  %43 = inttoptr i64 %42 to ptr
  br label %44

44:                                               ; preds = %60, %37
  %45 = phi i64 [ %61, %60 ], [ 0, %37 ]
  %46 = icmp slt i64 %45, 4
  br i1 %46, label %47, label %62

47:                                               ; preds = %44
  br label %48

48:                                               ; preds = %51, %47
  %49 = phi i64 [ %59, %51 ], [ 0, %47 ]
  %50 = icmp slt i64 %49, 5
  br i1 %50, label %51, label %60

51:                                               ; preds = %48
  %52 = mul nuw nsw i64 %49, 4
  %53 = add nuw nsw i64 %52, %45
  %54 = getelementptr inbounds nuw float, ptr @__constant_5x4xf32, i64 %53
  %55 = load float, ptr %54, align 4
  %56 = mul nuw nsw i64 %45, 5
  %57 = add nuw nsw i64 %56, %49
  %58 = getelementptr inbounds nuw float, ptr %43, i64 %57
  store float %55, ptr %58, align 4
  %59 = add i64 %49, 1
  br label %48

60:                                               ; preds = %48
  %61 = add i64 %45, 1
  br label %44

62:                                               ; preds = %44
  %63 = call ptr @malloc(i64 124)
  %64 = ptrtoint ptr %63 to i64
  %65 = add i64 %64, 63
  %66 = urem i64 %65, 64
  %67 = sub i64 %65, %66
  %68 = inttoptr i64 %67 to ptr
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %63, 0
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, ptr %68, 1
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 0, 2
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %71, i64 3, 3, 0
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, i64 5, 3, 1
  %74 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %73, i64 5, 4, 0
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %74, i64 1, 4, 1
  br label %76

76:                                               ; preds = %88, %62
  %77 = phi i64 [ %89, %88 ], [ 0, %62 ]
  %78 = icmp slt i64 %77, 3
  br i1 %78, label %79, label %90

79:                                               ; preds = %76
  br label %80

80:                                               ; preds = %83, %79
  %81 = phi i64 [ %87, %83 ], [ 0, %79 ]
  %82 = icmp slt i64 %81, 5
  br i1 %82, label %83, label %88

83:                                               ; preds = %80
  %84 = mul nuw nsw i64 %77, 5
  %85 = add nuw nsw i64 %84, %81
  %86 = getelementptr inbounds nuw float, ptr %68, i64 %85
  store float 0.000000e+00, ptr %86, align 4
  %87 = add i64 %81, 1
  br label %80

88:                                               ; preds = %80
  %89 = add i64 %77, 1
  br label %76

90:                                               ; preds = %76
  br label %91

91:                                               ; preds = %120, %90
  %92 = phi i64 [ %121, %120 ], [ 0, %90 ]
  %93 = icmp slt i64 %92, 3
  br i1 %93, label %94, label %122

94:                                               ; preds = %91
  br label %95

95:                                               ; preds = %118, %94
  %96 = phi i64 [ %119, %118 ], [ 0, %94 ]
  %97 = icmp slt i64 %96, 5
  br i1 %97, label %98, label %120

98:                                               ; preds = %95
  br label %99

99:                                               ; preds = %102, %98
  %100 = phi i64 [ %117, %102 ], [ 0, %98 ]
  %101 = icmp slt i64 %100, 4
  br i1 %101, label %102, label %118

102:                                              ; preds = %99
  %103 = mul nuw nsw i64 %92, 4
  %104 = add nuw nsw i64 %103, %100
  %105 = getelementptr inbounds nuw float, ptr %13, i64 %104
  %106 = load float, ptr %105, align 4
  %107 = mul nuw nsw i64 %100, 5
  %108 = add nuw nsw i64 %107, %96
  %109 = getelementptr inbounds nuw float, ptr %43, i64 %108
  %110 = load float, ptr %109, align 4
  %111 = mul nuw nsw i64 %92, 5
  %112 = add nuw nsw i64 %111, %96
  %113 = getelementptr inbounds nuw float, ptr %68, i64 %112
  %114 = load float, ptr %113, align 4
  %115 = fmul float %106, %110
  %116 = fadd float %114, %115
  store float %116, ptr %113, align 4
  %117 = add i64 %100, 1
  br label %99

118:                                              ; preds = %99
  %119 = add i64 %96, 1
  br label %95

120:                                              ; preds = %95
  %121 = add i64 %92, 1
  br label %91

122:                                              ; preds = %91
  br label %123

123:                                              ; preds = %139, %122
  %124 = phi i64 [ %140, %139 ], [ 0, %122 ]
  %125 = icmp slt i64 %124, 3
  br i1 %125, label %126, label %141

126:                                              ; preds = %123
  br label %127

127:                                              ; preds = %130, %126
  %128 = phi i64 [ %138, %130 ], [ 0, %126 ]
  %129 = icmp slt i64 %128, 5
  br i1 %129, label %130, label %139

130:                                              ; preds = %127
  %131 = mul nuw nsw i64 %124, 5
  %132 = add nuw nsw i64 %131, %128
  %133 = getelementptr inbounds nuw float, ptr %68, i64 %132
  %134 = load float, ptr %133, align 4
  %135 = getelementptr inbounds nuw float, ptr @__constant_5xf32, i64 %128
  %136 = load float, ptr %135, align 4
  %137 = fadd float %134, %136
  store float %137, ptr %133, align 4
  %138 = add i64 %128, 1
  br label %127

139:                                              ; preds = %127
  %140 = add i64 %124, 1
  br label %123

141:                                              ; preds = %123
  br label %142

142:                                              ; preds = %159, %141
  %143 = phi i64 [ %160, %159 ], [ 0, %141 ]
  %144 = icmp slt i64 %143, 3
  br i1 %144, label %145, label %161

145:                                              ; preds = %142
  br label %146

146:                                              ; preds = %149, %145
  %147 = phi i64 [ %158, %149 ], [ 0, %145 ]
  %148 = icmp slt i64 %147, 5
  br i1 %148, label %149, label %159

149:                                              ; preds = %146
  %150 = mul nuw nsw i64 %143, 5
  %151 = add nuw nsw i64 %150, %147
  %152 = getelementptr inbounds nuw float, ptr %68, i64 %151
  %153 = load float, ptr %152, align 4
  %154 = fcmp ult float %153, 0.000000e+00
  %155 = select i1 %154, float 0.000000e+00, float %153
  %156 = fcmp ugt float %155, 1.000000e+00
  %157 = select i1 %156, float 1.000000e+00, float %155
  store float %157, ptr %152, align 4
  %158 = add i64 %147, 1
  br label %146

159:                                              ; preds = %146
  %160 = add i64 %143, 1
  br label %142

161:                                              ; preds = %142
  call void @free(ptr %8)
  call void @free(ptr %38)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %75
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
