; ModuleID = 'gam1.ll'
source_filename = "OUT/src/gated_act_mul_gated_act_mul_metal__GatedActMul_v1_2d0f50e9.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

%"struct.metal::vec_sel<metal::bfloat, 4>::type" = type { [4 x i16] }
%"struct.metal::bfloat" = type { i16 }

; Function Attrs: convergent mustprogress norecurse nounwind
define dso_local spir_kernel void @GatedActMul_v1(ptr addrspace(1) noundef align 2 %0, ptr addrspace(1) noundef align 2 %1, ptr addrspace(1) noundef align 2 %2, ptr addrspace(1) noundef align 1 %3, ptr addrspace(1) noundef align 4 %4, ptr addrspace(1) noundef align 4 %5, ptr addrspace(1) noundef align 4 %6, ptr addrspace(2) noundef align 4 %7, ptr addrspace(2) noundef align 4 %8, ptr addrspace(2) noundef align 4 %9, ptr addrspace(2) noundef align 4 %10, ptr addrspace(2) noundef align 4 %11, ptr addrspace(2) noundef align 4 %12, ptr addrspace(2) noundef align 4 %13, ptr addrspace(2) noundef align 4 %14, ptr addrspace(2) noundef align 4 %15, ptr addrspace(2) noundef align 4 %16, ptr addrspace(2) noundef align 4 %17, i32 noundef %18, i32 noundef %19, i32 noundef %20, i32 noundef %21, i32 noundef %22, i32 noundef %23, i32 noundef %24, i32 noundef %25) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 !kernel_arg_name !9 {
  %.sroa.7 = alloca [2 x i8], align 2
  %.sroa.7806 = alloca [2 x i8], align 2
  %.sroa.7811 = alloca [2 x i8], align 2
  %.sroa.7816 = alloca [2 x i8], align 2
  %.sroa.5789 = alloca [2 x i8], align 2
  %.sroa.5786 = alloca [2 x i8], align 2
  %.sroa.5783 = alloca [2 x i8], align 2
  %.sroa.5792 = alloca [2 x i8], align 2
  %.sroa.5795 = alloca [2 x i8], align 2
  %.sroa.5 = alloca [2 x i8], align 2
  %.sroa.9 = alloca [2 x i8], align 2
  %.sroa.6821 = alloca [2 x i8], align 2
  %27 = alloca [4 x i8], align 1
  %28 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %.sroa.17 = alloca [2 x i8], align 2
  %.sroa.5798 = alloca [2 x i8], align 2
  %.sroa.5801 = alloca [2 x i8], align 2
  %.sroa.7844 = alloca [2 x i8], align 2
  %29 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %30 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %31 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %32 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %33 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %34 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %35 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %36 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %37 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %38 = alloca %"struct.metal::vec_sel<metal::bfloat, 4>::type", align 2
  %.sroa.6839 = alloca [2 x i8], align 2
  %.sroa.5836 = alloca [2 x i8], align 2
  %.sroa.6833 = alloca [2 x i8], align 2
  %.0.copyload = load i32, ptr addrspace(2) %17, align 4
  %39 = icmp ne i32 %18, 0
  %40 = zext i1 %39 to i8
  %41 = icmp ne i32 %19, 0
  %42 = zext i1 %41 to i8
  %43 = icmp ne i32 %20, 0
  %44 = zext i1 %43 to i8
  %45 = icmp ne i32 %23, 0
  %46 = zext i1 %45 to i8
  %47 = icmp ne i32 %24, 0
  %48 = zext i1 %47 to i8
  %49 = icmp ne i32 %25, 0
  %50 = zext i1 %49 to i8
  %51 = call spir_func i32 @_Z12get_group_idj(i32 noundef 0) #5
  %52 = call spir_func i32 @_Z12get_group_idj(i32 noundef 1) #5
  %53 = call spir_func i32 @_Z12get_local_idj(i32 noundef 0) #5
  %54 = call spir_func i32 @_Z22get_sub_group_local_idv() #6
  %55 = call spir_func i32 @_Z16get_sub_group_idv() #6
  %56 = call spir_func i32 @_Z18get_sub_group_sizev() #6
  %57 = call spir_func i32 @_Z18get_num_sub_groupsv() #6
  %58 = call spir_func i32 @_Z12get_group_idj(i32 noundef 2) #5
  %59 = insertelement <3 x i32> poison, i32 %51, i32 0
  %60 = insertelement <3 x i32> %59, i32 %52, i32 1
  %61 = insertelement <3 x i32> %60, i32 %58, i32 2
  %62 = call spir_func i32 @_Z14get_local_sizej(i32 noundef 0) #5
  %63 = call spir_func i32 @_Z14get_local_sizej(i32 noundef 1) #5
  %64 = call spir_func i32 @_Z14get_local_sizej(i32 noundef 2) #5
  %65 = insertelement <3 x i32> poison, i32 %62, i32 0
  %66 = insertelement <3 x i32> %65, i32 %63, i32 1
  %67 = insertelement <3 x i32> %66, i32 %64, i32 2
  %68 = call spir_func i32 @_Z14get_num_groupsj(i32 noundef 0) #5
  %69 = call spir_func i32 @_Z14get_num_groupsj(i32 noundef 1) #5
  %70 = call spir_func i32 @_Z14get_num_groupsj(i32 noundef 2) #5
  %71 = insertelement <3 x i32> poison, i32 %68, i32 0
  %72 = insertelement <3 x i32> %71, i32 %69, i32 1
  %73 = insertelement <3 x i32> %72, i32 %70, i32 2
  %74 = call spir_func i32 @_Z15get_global_sizej(i32 noundef 0) #5
  %75 = call spir_func i32 @_Z15get_global_sizej(i32 noundef 1) #5
  %76 = call spir_func i32 @_Z15get_global_sizej(i32 noundef 2) #5
  %77 = insertelement <3 x i32> poison, i32 %74, i32 0
  %78 = insertelement <3 x i32> %77, i32 %75, i32 1
  %79 = insertelement <3 x i32> %78, i32 %76, i32 2
  call void @llvm.lifetime.start.p0(ptr %27)
  call void @llvm.lifetime.start.p0(ptr %28)
  call void @llvm.lifetime.start.p0(ptr %.sroa.17)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5798)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5801)
  call void @llvm.lifetime.start.p0(ptr %.sroa.7844)
  call void @llvm.lifetime.start.p0(ptr %29)
  call void @llvm.lifetime.start.p0(ptr %30)
  call void @llvm.lifetime.start.p0(ptr %31)
  call void @llvm.lifetime.start.p0(ptr %32)
  call void @llvm.lifetime.start.p0(ptr %33)
  call void @llvm.lifetime.start.p0(ptr %34)
  call void @llvm.lifetime.start.p0(ptr %35)
  call void @llvm.lifetime.start.p0(ptr %36)
  call void @llvm.lifetime.start.p0(ptr %37)
  call void @llvm.lifetime.start.p0(ptr %38)
  call void @llvm.lifetime.start.p0(ptr %.sroa.6839)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5836)
  call void @llvm.lifetime.start.p0(ptr %.sroa.6833)
  %80 = mul i32 %51, 4
  %81 = mul i32 %80, 32
  %82 = mul i32 %81, 4
  %83 = mul i32 %53, 4
  %84 = add i32 %82, %83
  %85 = trunc i32 %54 to i16
  %86 = load i32, ptr addrspace(2) %7, align 4
  %87 = icmp ult i32 %84, %86
  br i1 %87, label %88, label %91

88:                                               ; preds = %26
  %89 = sub i32 %86, %84
  %90 = call spir_func i32 @_Z3minjj(i32 noundef 4, i32 noundef %89) #5
  br label %91

91:                                               ; preds = %88, %26
  %92 = phi i32 [ %90, %88 ], [ 0, %26 ]
  %93 = icmp eq i32 %92, 4
  %94 = zext i1 %93 to i8
  %95 = addrspacecast ptr %31 to ptr addrspace(4)
  br label %96

96:                                               ; preds = %98, %91
  %.0742 = phi i32 [ 0, %91 ], [ %100, %98 ]
  %97 = icmp slt i32 %.0742, 4
  br i1 %97, label %98, label %101

98:                                               ; preds = %96
  %99 = getelementptr inbounds [4 x i16], ptr addrspace(4) %95, i32 0, i32 %.0742
  store i16 0, ptr addrspace(4) %99, align 2
  %100 = add nsw i32 %.0742, 1
  br label %96, !llvm.loop !10

101:                                              ; preds = %96
  %102 = addrspacecast ptr %32 to ptr addrspace(4)
  br label %103

103:                                              ; preds = %105, %101
  %.0741 = phi i32 [ 0, %101 ], [ %107, %105 ]
  %104 = icmp slt i32 %.0741, 4
  br i1 %104, label %105, label %108

105:                                              ; preds = %103
  %106 = getelementptr inbounds [4 x i16], ptr addrspace(4) %102, i32 0, i32 %.0741
  store i16 0, ptr addrspace(4) %106, align 2
  %107 = add nsw i32 %.0741, 1
  br label %103, !llvm.loop !10

108:                                              ; preds = %103
  br i1 %41, label %109, label %115

109:                                              ; preds = %108
  %110 = load i32, ptr addrspace(2) %7, align 4
  %111 = mul i32 2, %110
  %112 = mul i32 %52, %111
  %113 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %0, i32 %112
  %114 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %113, i32 %110
  br label %124

115:                                              ; preds = %108
  %116 = load i32, ptr addrspace(2) %10, align 4
  %117 = mul i32 %52, %116
  %118 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %1, i32 %117
  %119 = load i32, ptr addrspace(2) %9, align 4
  %120 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %118, i32 %119
  %121 = load i32, ptr addrspace(2) %7, align 4
  %122 = mul i32 %52, %121
  %123 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %0, i32 %122
  br label %124

124:                                              ; preds = %115, %109
  %.0744 = phi ptr addrspace(1) [ %114, %109 ], [ %123, %115 ]
  %.0743 = phi ptr addrspace(1) [ %113, %109 ], [ %120, %115 ]
  br i1 %93, label %125, label %161

125:                                              ; preds = %124
  %126 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %.0743, i32 %84
  %127 = addrspacecast ptr %33 to ptr addrspace(4)
  %128 = addrspacecast ptr addrspace(1) %126 to ptr addrspace(4)
  br label %129

129:                                              ; preds = %131, %125
  %.0734 = phi i32 [ 0, %125 ], [ %135, %131 ]
  %130 = icmp slt i32 %.0734, 4
  br i1 %130, label %131, label %136

131:                                              ; preds = %129
  %132 = getelementptr inbounds [4 x i16], ptr addrspace(4) %128, i32 0, i32 %.0734
  %133 = load i16, ptr addrspace(4) %132, align 2
  %134 = getelementptr inbounds [4 x i16], ptr addrspace(4) %127, i32 0, i32 %.0734
  store i16 %133, ptr addrspace(4) %134, align 2
  %135 = add nsw i32 %.0734, 1
  br label %129, !llvm.loop !12

136:                                              ; preds = %138, %129
  %.0737 = phi i32 [ %142, %138 ], [ 0, %129 ]
  %137 = icmp slt i32 %.0737, 4
  br i1 %137, label %138, label %143

138:                                              ; preds = %136
  %139 = getelementptr inbounds [4 x i16], ptr addrspace(4) %127, i32 0, i32 %.0737
  %140 = load i16, ptr addrspace(4) %139, align 2
  %141 = getelementptr inbounds [4 x i16], ptr addrspace(4) %95, i32 0, i32 %.0737
  store i16 %140, ptr addrspace(4) %141, align 2
  %142 = add nsw i32 %.0737, 1
  br label %136, !llvm.loop !13

143:                                              ; preds = %136
  %144 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %.0744, i32 %84
  %145 = addrspacecast ptr %34 to ptr addrspace(4)
  %146 = addrspacecast ptr addrspace(1) %144 to ptr addrspace(4)
  br label %147

147:                                              ; preds = %149, %143
  %.0733 = phi i32 [ 0, %143 ], [ %153, %149 ]
  %148 = icmp slt i32 %.0733, 4
  br i1 %148, label %149, label %154

149:                                              ; preds = %147
  %150 = getelementptr inbounds [4 x i16], ptr addrspace(4) %146, i32 0, i32 %.0733
  %151 = load i16, ptr addrspace(4) %150, align 2
  %152 = getelementptr inbounds [4 x i16], ptr addrspace(4) %145, i32 0, i32 %.0733
  store i16 %151, ptr addrspace(4) %152, align 2
  %153 = add nsw i32 %.0733, 1
  br label %147, !llvm.loop !12

154:                                              ; preds = %156, %147
  %.0738 = phi i32 [ %160, %156 ], [ 0, %147 ]
  %155 = icmp slt i32 %.0738, 4
  br i1 %155, label %156, label %178

156:                                              ; preds = %154
  %157 = getelementptr inbounds [4 x i16], ptr addrspace(4) %145, i32 0, i32 %.0738
  %158 = load i16, ptr addrspace(4) %157, align 2
  %159 = getelementptr inbounds [4 x i16], ptr addrspace(4) %102, i32 0, i32 %.0738
  store i16 %158, ptr addrspace(4) %159, align 2
  %160 = add nsw i32 %.0738, 1
  br label %154, !llvm.loop !13

161:                                              ; preds = %176, %124
  %.0745 = phi i32 [ %177, %176 ], [ 0, %124 ]
  %162 = icmp ult i32 %.0745, %92
  br i1 %162, label %163, label %178

163:                                              ; preds = %161
  %164 = add i32 %84, %.0745
  %165 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %.0743, i32 %164
  %166 = addrspacecast ptr addrspace(1) %165 to ptr addrspace(4)
  %.sroa.0125.0.copyload = load i16, ptr addrspace(4) %166, align 2
  %167 = getelementptr inbounds [4 x i16], ptr addrspace(4) %95, i32 0, i32 %.0745
  %168 = icmp ne ptr addrspace(4) %167, null
  br i1 %168, label %169, label %170

169:                                              ; preds = %163
  store i16 %.sroa.0125.0.copyload, ptr addrspace(4) %167, align 2
  br label %170

170:                                              ; preds = %169, %163
  %171 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %.0744, i32 %164
  %172 = addrspacecast ptr addrspace(1) %171 to ptr addrspace(4)
  %.sroa.0123.0.copyload = load i16, ptr addrspace(4) %172, align 2
  %173 = getelementptr inbounds [4 x i16], ptr addrspace(4) %102, i32 0, i32 %.0745
  %174 = icmp ne ptr addrspace(4) %173, null
  br i1 %174, label %175, label %176

175:                                              ; preds = %170
  store i16 %.sroa.0123.0.copyload, ptr addrspace(4) %173, align 2
  br label %176

176:                                              ; preds = %175, %170
  %177 = add i32 %.0745, 1
  br label %161, !llvm.loop !14

178:                                              ; preds = %161, %154
  br i1 %47, label %179, label %236

179:                                              ; preds = %178
  %180 = addrspacecast ptr %35 to ptr addrspace(4)
  %181 = addrspacecast ptr %36 to ptr addrspace(4)
  br label %182

182:                                              ; preds = %184, %179
  %.0732 = phi i32 [ 0, %179 ], [ %188, %184 ]
  %183 = icmp slt i32 %.0732, 4
  br i1 %183, label %184, label %189

184:                                              ; preds = %182
  %185 = getelementptr inbounds [4 x i16], ptr addrspace(4) %102, i32 0, i32 %.0732
  %186 = load i16, ptr addrspace(4) %185, align 2
  %187 = getelementptr inbounds [4 x i16], ptr addrspace(4) %181, i32 0, i32 %.0732
  store i16 %186, ptr addrspace(4) %187, align 2
  %188 = add nsw i32 %.0732, 1
  br label %182, !llvm.loop !12

189:                                              ; preds = %182
  %190 = addrspacecast ptr %29 to ptr addrspace(4)
  br label %191

191:                                              ; preds = %193, %189
  %.0736 = phi i32 [ 0, %189 ], [ %197, %193 ]
  %192 = icmp slt i32 %.0736, 4
  br i1 %192, label %193, label %198

193:                                              ; preds = %191
  %194 = getelementptr inbounds [4 x i16], ptr addrspace(4) %181, i32 0, i32 %.0736
  %195 = load i16, ptr addrspace(4) %194, align 2
  %196 = getelementptr inbounds [4 x i16], ptr addrspace(4) %190, i32 0, i32 %.0736
  store i16 %195, ptr addrspace(4) %196, align 2
  %197 = add nsw i32 %.0736, 1
  br label %191, !llvm.loop !12

198:                                              ; preds = %205, %191
  %.0718 = phi i32 [ %211, %205 ], [ 0, %191 ]
  %.0717 = phi <4 x float> [ %210, %205 ], [ undef, %191 ]
  %199 = icmp slt i32 %.0718, 4
  br i1 %199, label %200, label %212

200:                                              ; preds = %198
  %201 = getelementptr inbounds [4 x i16], ptr addrspace(4) %190, i32 0, i32 %.0718
  %202 = icmp ne ptr addrspace(4) %201, null
  br i1 %202, label %203, label %205

203:                                              ; preds = %200
  %204 = load i16, ptr addrspace(4) %201, align 2
  br label %205

205:                                              ; preds = %203, %200
  %206 = phi i16 [ %204, %203 ], [ 0, %200 ]
  %207 = zext i16 %206 to i32
  %208 = shl i32 %207, 16
  %209 = bitcast i32 %208 to float
  %210 = insertelement <4 x float> %.0717, float %209, i32 %.0718
  %211 = add nsw i32 %.0718, 1
  br label %198, !llvm.loop !15

212:                                              ; preds = %198
  %213 = load float, ptr addrspace(2) %13, align 4
  %214 = load float, ptr addrspace(2) %14, align 4
  %215 = call spir_func <4 x float> @_Z5clampDv4_fff(<4 x float> noundef %.0717, float noundef %213, float noundef %214) #5
  br label %216

216:                                              ; preds = %218, %212
  %.0730 = phi i32 [ 0, %212 ], [ %228, %218 ]
  %217 = icmp slt i32 %.0730, 4
  br i1 %217, label %218, label %229

218:                                              ; preds = %216
  %219 = extractelement <4 x float> %215, i32 %.0730
  %220 = bitcast float %219 to i32
  %221 = add i32 %220, 32767
  %222 = lshr i32 %220, 16
  %223 = and i32 %222, 1
  %224 = add i32 %221, %223
  %225 = lshr i32 %224, 16
  %226 = trunc i32 %225 to i16
  %227 = getelementptr inbounds [4 x i16], ptr addrspace(4) %180, i32 0, i32 %.0730
  store i16 %226, ptr addrspace(4) %227, align 2
  %228 = add nsw i32 %.0730, 1
  br label %216, !llvm.loop !16

229:                                              ; preds = %231, %216
  %.0739 = phi i32 [ %235, %231 ], [ 0, %216 ]
  %230 = icmp slt i32 %.0739, 4
  br i1 %230, label %231, label %236

231:                                              ; preds = %229
  %232 = getelementptr inbounds [4 x i16], ptr addrspace(4) %180, i32 0, i32 %.0739
  %233 = load i16, ptr addrspace(4) %232, align 2
  %234 = getelementptr inbounds [4 x i16], ptr addrspace(4) %102, i32 0, i32 %.0739
  store i16 %233, ptr addrspace(4) %234, align 2
  %235 = add nsw i32 %.0739, 1
  br label %229, !llvm.loop !13

236:                                              ; preds = %229, %178
  br i1 %49, label %237, label %294

237:                                              ; preds = %236
  %238 = addrspacecast ptr %37 to ptr addrspace(4)
  %239 = addrspacecast ptr %38 to ptr addrspace(4)
  br label %240

240:                                              ; preds = %242, %237
  %.0731 = phi i32 [ 0, %237 ], [ %246, %242 ]
  %241 = icmp slt i32 %.0731, 4
  br i1 %241, label %242, label %247

242:                                              ; preds = %240
  %243 = getelementptr inbounds [4 x i16], ptr addrspace(4) %95, i32 0, i32 %.0731
  %244 = load i16, ptr addrspace(4) %243, align 2
  %245 = getelementptr inbounds [4 x i16], ptr addrspace(4) %239, i32 0, i32 %.0731
  store i16 %244, ptr addrspace(4) %245, align 2
  %246 = add nsw i32 %.0731, 1
  br label %240, !llvm.loop !12

247:                                              ; preds = %240
  %248 = addrspacecast ptr %30 to ptr addrspace(4)
  br label %249

249:                                              ; preds = %251, %247
  %.0735 = phi i32 [ 0, %247 ], [ %255, %251 ]
  %250 = icmp slt i32 %.0735, 4
  br i1 %250, label %251, label %256

251:                                              ; preds = %249
  %252 = getelementptr inbounds [4 x i16], ptr addrspace(4) %239, i32 0, i32 %.0735
  %253 = load i16, ptr addrspace(4) %252, align 2
  %254 = getelementptr inbounds [4 x i16], ptr addrspace(4) %248, i32 0, i32 %.0735
  store i16 %253, ptr addrspace(4) %254, align 2
  %255 = add nsw i32 %.0735, 1
  br label %249, !llvm.loop !12

256:                                              ; preds = %263, %249
  %.0716 = phi i32 [ %269, %263 ], [ 0, %249 ]
  %.0715 = phi <4 x float> [ %268, %263 ], [ undef, %249 ]
  %257 = icmp slt i32 %.0716, 4
  br i1 %257, label %258, label %270

258:                                              ; preds = %256
  %259 = getelementptr inbounds [4 x i16], ptr addrspace(4) %248, i32 0, i32 %.0716
  %260 = icmp ne ptr addrspace(4) %259, null
  br i1 %260, label %261, label %263

261:                                              ; preds = %258
  %262 = load i16, ptr addrspace(4) %259, align 2
  br label %263

263:                                              ; preds = %261, %258
  %264 = phi i16 [ %262, %261 ], [ 0, %258 ]
  %265 = zext i16 %264 to i32
  %266 = shl i32 %265, 16
  %267 = bitcast i32 %266 to float
  %268 = insertelement <4 x float> %.0715, float %267, i32 %.0716
  %269 = add nsw i32 %.0716, 1
  br label %256, !llvm.loop !15

270:                                              ; preds = %256
  %271 = load float, ptr addrspace(2) %15, align 4
  %272 = load float, ptr addrspace(2) %16, align 4
  %273 = call spir_func <4 x float> @_Z5clampDv4_fff(<4 x float> noundef %.0715, float noundef %271, float noundef %272) #5
  br label %274

274:                                              ; preds = %276, %270
  %.0729 = phi i32 [ 0, %270 ], [ %286, %276 ]
  %275 = icmp slt i32 %.0729, 4
  br i1 %275, label %276, label %287

276:                                              ; preds = %274
  %277 = extractelement <4 x float> %273, i32 %.0729
  %278 = bitcast float %277 to i32
  %279 = add i32 %278, 32767
  %280 = lshr i32 %278, 16
  %281 = and i32 %280, 1
  %282 = add i32 %279, %281
  %283 = lshr i32 %282, 16
  %284 = trunc i32 %283 to i16
  %285 = getelementptr inbounds [4 x i16], ptr addrspace(4) %238, i32 0, i32 %.0729
  store i16 %284, ptr addrspace(4) %285, align 2
  %286 = add nsw i32 %.0729, 1
  br label %274, !llvm.loop !16

287:                                              ; preds = %289, %274
  %.0740 = phi i32 [ %293, %289 ], [ 0, %274 ]
  %288 = icmp slt i32 %.0740, 4
  br i1 %288, label %289, label %294

289:                                              ; preds = %287
  %290 = getelementptr inbounds [4 x i16], ptr addrspace(4) %238, i32 0, i32 %.0740
  %291 = load i16, ptr addrspace(4) %290, align 2
  %292 = getelementptr inbounds [4 x i16], ptr addrspace(4) %95, i32 0, i32 %.0740
  store i16 %291, ptr addrspace(4) %292, align 2
  %293 = add nsw i32 %.0740, 1
  br label %287, !llvm.loop !13

294:                                              ; preds = %287, %236
  %295 = add i32 %.0.copyload, -1
  %296 = icmp ult i32 %295, 2
  %297 = select i1 %296, i1 true, i1 %43
  %298 = zext i1 %297 to i8
  %299 = udiv i32 %84, 32
  %300 = add i32 %299, 1
  %301 = mul i32 %300, 32
  %302 = load i32, ptr addrspace(2) %7, align 4
  %303 = icmp ule i32 %301, %302
  %304 = zext i1 %303 to i8
  br label %305

305:                                              ; preds = %510, %294
  %.sroa.0845.0 = phi ptr addrspace(4) [ undef, %294 ], [ %.sroa.0845.1, %510 ]
  %.sroa.10.0 = phi i16 [ undef, %294 ], [ %.sroa.10.1, %510 ]
  %.0750 = phi i32 [ 0, %294 ], [ %511, %510 ]
  %.0746 = phi <4 x float> [ zeroinitializer, %294 ], [ %.3749, %510 ]
  %306 = icmp ult i32 %.0750, 4
  br i1 %306, label %307, label %512

307:                                              ; preds = %305
  %308 = icmp ult i32 %.0750, %92
  %.not = xor i1 %297, true
  %brmerge = select i1 %.not, i1 true, i1 %303
  %or.cond = select i1 %308, i1 %brmerge, i1 false
  br i1 %or.cond, label %309, label %510

309:                                              ; preds = %307
  br i1 %45, label %310, label %365

310:                                              ; preds = %309
  %311 = load i32, ptr addrspace(2) %11, align 4
  %312 = icmp eq i32 %311, 0
  br i1 %312, label %313, label %365

313:                                              ; preds = %310
  %314 = getelementptr inbounds [4 x i16], ptr addrspace(4) %102, i32 0, i32 %.0750
  %315 = load float, ptr addrspace(2) %12, align 4
  call void @llvm.lifetime.start.p0(ptr %.sroa.6821)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.6821, ptr align 2 %.sroa.6839, i64 2, i1 false)
  %316 = icmp ne ptr addrspace(4) %314, null
  br i1 %316, label %317, label %_Z19activate_silu_alphaIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_f.exit.i

317:                                              ; preds = %313
  %318 = load i16, ptr addrspace(4) %314, align 2
  br label %_Z19activate_silu_alphaIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_f.exit.i

_Z19activate_silu_alphaIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_f.exit.i: ; preds = %317, %313
  %319 = phi i16 [ %318, %317 ], [ 0, %313 ]
  %320 = zext i16 %319 to i32
  %321 = shl i32 %320, 16
  %322 = bitcast i32 %321 to float
  %323 = call spir_func float @_Z4fabsf(float noundef %322) #5
  %324 = fneg float %323
  %325 = fmul float %324, %315
  %326 = call spir_func float @_Z10native_expf(float noundef %325) #5
  %327 = fadd float 1.000000e+00, %326
  %328 = fdiv float 1.000000e+00, %327, !fpmath !17
  %329 = fcmp olt float %322, 0.000000e+00
  %330 = fsub float 1.000000e+00, %328
  %331 = fmul float %330, %322
  %332 = fmul float %328, %322
  %333 = select i1 %329, float %331, float %332
  %334 = bitcast float %333 to i32
  %335 = add i32 %334, 32767
  %336 = lshr i32 %334, 16
  %337 = and i32 %336, 1
  %338 = add i32 %335, %337
  %339 = lshr i32 %338, 16
  %340 = trunc i32 %339 to i16
  call void @llvm.lifetime.end.p0(ptr %.sroa.6821)
  %341 = getelementptr inbounds [4 x i16], ptr addrspace(4) %95, i32 0, i32 %.0750
  %342 = icmp ne ptr addrspace(4) %341, null
  br i1 %342, label %343, label %345

343:                                              ; preds = %_Z19activate_silu_alphaIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_f.exit.i
  %344 = load i16, ptr addrspace(4) %341, align 2
  br label %345

345:                                              ; preds = %343, %_Z19activate_silu_alphaIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_f.exit.i
  %346 = phi i16 [ %344, %343 ], [ 0, %_Z19activate_silu_alphaIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_f.exit.i ]
  %347 = zext i16 %346 to i32
  %348 = shl i32 %347, 16
  %349 = bitcast i32 %348 to float
  %350 = zext i16 %340 to i32
  %351 = shl i32 %350, 16
  %352 = bitcast i32 %351 to float
  %353 = fmul float %349, %352
  %354 = bitcast float %353 to i32
  %355 = add i32 %354, 32767
  %356 = lshr i32 %354, 16
  %357 = and i32 %356, 1
  %358 = add i32 %355, %357
  %359 = lshr i32 %358, 16
  %360 = trunc i32 %359 to i16
  %361 = zext i16 %360 to i32
  %362 = shl i32 %361, 16
  %363 = bitcast i32 %362 to float
  %364 = insertelement <4 x float> %.0746, float %363, i32 %.0750
  br label %510

365:                                              ; preds = %310, %309
  %366 = getelementptr inbounds [4 x i16], ptr addrspace(4) %95, i32 0, i32 %.0750
  %367 = getelementptr inbounds [4 x i16], ptr addrspace(4) %102, i32 0, i32 %.0750
  %368 = load i32, ptr addrspace(2) %11, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.5801, ptr align 2 %.sroa.6833, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.7844, ptr align 2 %.sroa.5836, i64 2, i1 false)
  %369 = icmp ne ptr addrspace(4) %366, null
  br i1 %369, label %370, label %372

370:                                              ; preds = %365
  %371 = load i16, ptr addrspace(4) %366, align 2
  br label %372

372:                                              ; preds = %370, %365
  %373 = phi i16 [ %371, %370 ], [ 0, %365 ]
  %374 = zext i16 %373 to i32
  %375 = shl i32 %374, 16
  %376 = bitcast i32 %375 to float
  %.sroa.5801.6..sroa_cast = addrspacecast ptr %.sroa.5801 to ptr addrspace(4)
  call void @llvm.memcpy.p0.p4.i32(ptr align 2 %.sroa.5798, ptr addrspace(4) align 2 %.sroa.5801.6..sroa_cast, i32 2, i1 false)
  call void @llvm.lifetime.start.p0(ptr %.sroa.9)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5783)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5792)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5795)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.9, ptr align 2 %.sroa.5798, i64 2, i1 false)
  switch i32 %368, label %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i [
    i32 0, label %377
    i32 1, label %402
    i32 2, label %425
    i32 3, label %467
    i32 4, label %468
  ]

377:                                              ; preds = %372
  %.sroa.9.0..sroa_cast780 = addrspacecast ptr %.sroa.9 to ptr addrspace(4)
  call void @llvm.memcpy.p0.p4.i32(ptr align 2 %.sroa.5783, ptr addrspace(4) align 2 %.sroa.9.0..sroa_cast780, i32 2, i1 false)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5786)
  call void @llvm.lifetime.start.p0(ptr %.sroa.5789)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.5786, ptr align 2 %.sroa.5783, i64 2, i1 false)
  %.sroa.5786.0..sroa_cast = addrspacecast ptr %.sroa.5786 to ptr addrspace(4)
  call void @llvm.memcpy.p0.p4.i32(ptr align 2 %.sroa.5789, ptr addrspace(4) align 2 %.sroa.5786.0..sroa_cast, i32 2, i1 false)
  call void @llvm.lifetime.start.p0(ptr %.sroa.7816)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.7816, ptr align 2 %.sroa.5789, i64 2, i1 false)
  %378 = icmp ne ptr addrspace(4) %367, null
  br i1 %378, label %379, label %_Z13activate_siluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i

379:                                              ; preds = %377
  %380 = load i16, ptr addrspace(4) %367, align 2
  br label %_Z13activate_siluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i

_Z13activate_siluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i: ; preds = %379, %377
  %381 = phi i16 [ %380, %379 ], [ 0, %377 ]
  %382 = zext i16 %381 to i32
  %383 = shl i32 %382, 16
  %384 = bitcast i32 %383 to float
  %385 = call spir_func float @_Z4fabsf(float noundef %384) #5
  %386 = fneg float %385
  %387 = call spir_func float @_Z10native_expf(float noundef %386) #5
  %388 = fadd float 1.000000e+00, %387
  %389 = fdiv float 1.000000e+00, %388, !fpmath !17
  %390 = fcmp olt float %384, 0.000000e+00
  %391 = fsub float 1.000000e+00, %389
  %392 = fmul float %391, %384
  %393 = fmul float %389, %384
  %394 = select i1 %390, float %392, float %393
  %395 = bitcast float %394 to i32
  %396 = add i32 %395, 32767
  %397 = lshr i32 %395, 16
  %398 = and i32 %397, 1
  %399 = add i32 %396, %398
  %400 = lshr i32 %399, 16
  %401 = trunc i32 %400 to i16
  call void @llvm.lifetime.end.p0(ptr %.sroa.7816)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5786)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5789)
  br label %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i

402:                                              ; preds = %372
  %.sroa.9.0..sroa_cast778 = addrspacecast ptr %.sroa.9 to ptr addrspace(4)
  call void @llvm.memcpy.p0.p4.i32(ptr align 2 %.sroa.5792, ptr addrspace(4) align 2 %.sroa.9.0..sroa_cast778, i32 2, i1 false)
  call void @llvm.lifetime.start.p0(ptr %.sroa.7811)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.7811, ptr align 2 %.sroa.5792, i64 2, i1 false)
  %403 = icmp ne ptr addrspace(4) %367, null
  br i1 %403, label %404, label %_Z13activate_geluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i

404:                                              ; preds = %402
  %405 = load i16, ptr addrspace(4) %367, align 2
  br label %_Z13activate_geluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i

_Z13activate_geluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i: ; preds = %404, %402
  %406 = phi i16 [ %405, %404 ], [ 0, %402 ]
  %407 = zext i16 %406 to i32
  %408 = shl i32 %407, 16
  %409 = bitcast i32 %408 to float
  %410 = fmul float 5.000000e-01, %409
  %411 = fmul float 4.471500e-02, %409
  %412 = fmul float %411, %409
  %413 = call float @llvm.fmuladd.f32(float %412, float %409, float %409)
  %414 = fmul float f0x3F4C422A, %413
  %415 = call spir_func float @_Z4tanhf(float noundef %414) #5
  %416 = fadd float 1.000000e+00, %415
  %417 = fmul float %410, %416
  %418 = bitcast float %417 to i32
  %419 = add i32 %418, 32767
  %420 = lshr i32 %418, 16
  %421 = and i32 %420, 1
  %422 = add i32 %419, %421
  %423 = lshr i32 %422, 16
  %424 = trunc i32 %423 to i16
  call void @llvm.lifetime.end.p0(ptr %.sroa.7811)
  br label %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i

425:                                              ; preds = %372
  %.sroa.9.0..sroa_cast776 = addrspacecast ptr %.sroa.9 to ptr addrspace(4)
  call void @llvm.memcpy.p0.p4.i32(ptr align 2 %.sroa.5795, ptr addrspace(4) align 2 %.sroa.9.0..sroa_cast776, i32 2, i1 false)
  call void @llvm.lifetime.start.p0(ptr %.sroa.7806)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.7806, ptr align 2 %.sroa.5795, i64 2, i1 false)
  %426 = icmp ne ptr addrspace(4) %367, null
  br i1 %426, label %427, label %429

427:                                              ; preds = %425
  %428 = load i16, ptr addrspace(4) %367, align 2
  br label %429

429:                                              ; preds = %427, %425
  %430 = phi i16 [ %428, %427 ], [ 0, %425 ]
  %431 = zext i16 %430 to i32
  %432 = shl i32 %431, 16
  %433 = bitcast i32 %432 to float
  %434 = fmul float 5.000000e-01, %433
  %435 = fmul float f0x3F3504F3, %433
  %436 = call spir_func float @_Z4fabsf(float noundef %435) #5
  %437 = fmul float %435, %435
  %438 = fcmp ogt float %436, f0x3F6D8000
  br i1 %438, label %439, label %451

439:                                              ; preds = %429
  %440 = call spir_func float @_Z3fmafff(float noundef f0xB7910000, float noundef %436, float noundef f0x39C8E7D9) #5
  %441 = call spir_func float @_Z3fmafff(float noundef f0xBB7E8A1C, float noundef %436, float noundef f0x3CC6B1A1) #5
  %442 = call spir_func float @_Z3fmafff(float noundef %440, float noundef %437, float noundef %441) #5
  %443 = call spir_func float @_Z3fmafff(float noundef %442, float noundef %436, float noundef f0xBDDAAE5C) #5
  %444 = call spir_func float @_Z3fmafff(float noundef %443, float noundef %436, float noundef f0xBF228550) #5
  %445 = call spir_func float @_Z3fmafff(float noundef %444, float noundef %436, float noundef f0xBE03CE86) #5
  %446 = fneg float %436
  %447 = call spir_func float @_Z3fmafff(float noundef %445, float noundef %436, float noundef %446) #5
  %448 = call spir_func float @_Z3expf(float noundef %447) #5
  %449 = fsub float 1.000000e+00, %448
  %450 = call spir_func float @_Z8copysignff(float noundef %449, float noundef %435) #5
  br label %_Z19activate_gelu_exactIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i

451:                                              ; preds = %429
  %452 = call spir_func float @_Z3fmafff(float noundef f0xBA1C7000, float noundef %437, float noundef f0x3BA38D2C) #5
  %453 = call spir_func float @_Z3fmafff(float noundef %452, float noundef %437, float noundef f0xBCDB48D9) #5
  %454 = call spir_func float @_Z3fmafff(float noundef %453, float noundef %437, float noundef f0x3DE70E22) #5
  %455 = call spir_func float @_Z3fmafff(float noundef %454, float noundef %437, float noundef f0xBEC09380) #5
  %456 = call spir_func float @_Z3fmafff(float noundef %455, float noundef %437, float noundef f0x3E0375D4) #5
  %457 = call spir_func float @_Z3fmafff(float noundef %456, float noundef %435, float noundef %435) #5
  br label %_Z19activate_gelu_exactIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i

_Z19activate_gelu_exactIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i: ; preds = %451, %439
  %.0 = phi float [ %450, %439 ], [ %457, %451 ]
  %458 = fadd float 1.000000e+00, %.0
  %459 = fmul float %434, %458
  %460 = bitcast float %459 to i32
  %461 = add i32 %460, 32767
  %462 = lshr i32 %460, 16
  %463 = and i32 %462, 1
  %464 = add i32 %461, %463
  %465 = lshr i32 %464, 16
  %466 = trunc i32 %465 to i16
  call void @llvm.lifetime.end.p0(ptr %.sroa.7806)
  br label %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i

467:                                              ; preds = %372
  %.sroa.9.0..sroa_cast = addrspacecast ptr %.sroa.9 to ptr addrspace(4)
  call void @llvm.memcpy.p0.p4.i32(ptr align 2 %.sroa.17, ptr addrspace(4) align 2 %.sroa.9.0..sroa_cast, i32 2, i1 false)
  br label %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i

468:                                              ; preds = %372
  %.sroa.9.6..sroa_cast = addrspacecast ptr %.sroa.9 to ptr addrspace(4)
  call void @llvm.memcpy.p0.p4.i32(ptr align 2 %.sroa.5, ptr addrspace(4) align 2 %.sroa.9.6..sroa_cast, i32 2, i1 false)
  call void @llvm.lifetime.start.p0(ptr %.sroa.7)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %.sroa.7, ptr align 2 %.sroa.5, i64 2, i1 false)
  %469 = icmp ne ptr addrspace(4) %367, null
  br i1 %469, label %470, label %472

470:                                              ; preds = %468
  %471 = load i16, ptr addrspace(4) %367, align 2
  br label %472

472:                                              ; preds = %470, %468
  %473 = phi i16 [ %471, %470 ], [ 0, %468 ]
  %474 = zext i16 %473 to i32
  %475 = shl i32 %474, 16
  %476 = bitcast i32 %475 to float
  %477 = fcmp ogt float %476, 2.000000e+01
  br i1 %477, label %478, label %479

478:                                              ; preds = %472
  %.sroa.7.0..sroa_cast = addrspacecast ptr %.sroa.7 to ptr addrspace(4)
  call void @llvm.memcpy.p0.p4.i32(ptr align 2 %.sroa.17, ptr addrspace(4) align 2 %.sroa.7.0..sroa_cast, i32 2, i1 false)
  br label %_Z17activate_softplusIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i

479:                                              ; preds = %472
  %480 = call spir_func float @_Z10native_expf(float noundef %476) #5
  %481 = fadd float 1.000000e+00, %480
  %482 = call spir_func float @_Z10native_logf(float noundef %481) #5
  %483 = bitcast float %482 to i32
  %484 = add i32 %483, 32767
  %485 = lshr i32 %483, 16
  %486 = and i32 %485, 1
  %487 = add i32 %484, %486
  %488 = lshr i32 %487, 16
  %489 = trunc i32 %488 to i16
  br label %_Z17activate_softplusIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i

_Z17activate_softplusIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i: ; preds = %479, %478
  %.sroa.0845.2 = phi ptr addrspace(4) [ %367, %478 ], [ null, %479 ]
  %.sroa.10.2 = phi i16 [ 0, %478 ], [ %489, %479 ]
  call void @llvm.lifetime.end.p0(ptr %.sroa.7)
  br label %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i

_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i: ; preds = %_Z17activate_softplusIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i, %467, %_Z19activate_gelu_exactIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i, %_Z13activate_geluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i, %_Z13activate_siluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i, %372
  %.sroa.0845.3 = phi ptr addrspace(4) [ %.sroa.0845.0, %372 ], [ null, %_Z13activate_siluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i ], [ null, %_Z13activate_geluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i ], [ null, %_Z19activate_gelu_exactIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i ], [ %367, %467 ], [ %.sroa.0845.2, %_Z17activate_softplusIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i ]
  %.sroa.10.3 = phi i16 [ %.sroa.10.0, %372 ], [ %401, %_Z13activate_siluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i ], [ %424, %_Z13activate_geluIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i ], [ %466, %_Z19activate_gelu_exactIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i ], [ 0, %467 ], [ %.sroa.10.2, %_Z17activate_softplusIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_.exit.i.i ]
  call void @llvm.lifetime.end.p0(ptr %.sroa.9)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5783)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5792)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5795)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5)
  %490 = icmp ne ptr addrspace(4) %.sroa.0845.3, null
  br i1 %490, label %491, label %493

491:                                              ; preds = %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i
  %492 = load i16, ptr addrspace(4) %.sroa.0845.3, align 2
  br label %493

493:                                              ; preds = %491, %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i
  %494 = phi i16 [ %492, %491 ], [ %.sroa.10.3, %_Z8activateIN5metal7vec_selINS0_6bfloatELi4ELb0EE4type3refEET_S6_N3uzu15activation_type14ActivationTypeE.exit.i ]
  %495 = zext i16 %494 to i32
  %496 = shl i32 %495, 16
  %497 = bitcast i32 %496 to float
  %498 = fmul float %376, %497
  %499 = bitcast float %498 to i32
  %500 = add i32 %499, 32767
  %501 = lshr i32 %499, 16
  %502 = and i32 %501, 1
  %503 = add i32 %500, %502
  %504 = lshr i32 %503, 16
  %505 = trunc i32 %504 to i16
  %506 = zext i16 %505 to i32
  %507 = shl i32 %506, 16
  %508 = bitcast i32 %507 to float
  %509 = insertelement <4 x float> %.0746, float %508, i32 %.0750
  br label %510

510:                                              ; preds = %493, %345, %307
  %.sroa.0845.1 = phi ptr addrspace(4) [ %.sroa.0845.0, %307 ], [ %.sroa.0845.3, %493 ], [ %.sroa.0845.0, %345 ]
  %.sroa.10.1 = phi i16 [ %.sroa.10.0, %307 ], [ %.sroa.10.3, %493 ], [ %.sroa.10.0, %345 ]
  %.3749 = phi <4 x float> [ %.0746, %307 ], [ %509, %493 ], [ %364, %345 ]
  %511 = add i32 %.0750, 1
  br label %305, !llvm.loop !18

512:                                              ; preds = %305
  br i1 %297, label %513, label %566

513:                                              ; preds = %512
  br i1 %303, label %514, label %519

514:                                              ; preds = %513
  %515 = getelementptr inbounds nuw i32, ptr addrspace(1) %6, i32 %84
  %516 = load <4 x i32>, ptr addrspace(1) %515, align 16
  %517 = sitofp <4 x i32> %516 to <4 x float>
  %518 = fmul <4 x float> %.0746, %517
  br label %519

519:                                              ; preds = %514, %513
  %.2748 = phi <4 x float> [ %518, %514 ], [ %.0746, %513 ]
  br label %520

520:                                              ; preds = %539, %519
  %.0726 = phi i16 [ 1, %519 ], [ %541, %539 ]
  %.0724 = phi <4 x float> [ %.2748, %519 ], [ %.2, %539 ]
  %521 = zext i16 %.0726 to i32
  %522 = icmp slt i32 %521, 4
  br i1 %522, label %523, label %542

523:                                              ; preds = %537, %520
  %.0727 = phi i16 [ %538, %537 ], [ 0, %520 ]
  %.2 = phi <4 x float> [ %.3, %537 ], [ %.0724, %520 ]
  %524 = zext i16 %.0727 to i32
  %525 = icmp slt i32 %524, 4
  br i1 %525, label %526, label %539

526:                                              ; preds = %523
  %527 = and i32 %524, %521
  %528 = icmp ne i32 %527, 0
  br i1 %528, label %537, label %529

529:                                              ; preds = %526
  %530 = extractelement <4 x float> %.2, i16 %.0727
  %531 = add nsw i32 %524, %521
  %532 = extractelement <4 x float> %.2, i32 %531
  %533 = fadd float %532, %530
  %534 = insertelement <4 x float> %.2, float %533, i16 %.0727
  %535 = fsub float %530, %532
  %536 = insertelement <4 x float> %534, float %535, i32 %531
  br label %537

537:                                              ; preds = %529, %526
  %.3 = phi <4 x float> [ %.2, %526 ], [ %536, %529 ]
  %538 = add i16 %.0727, 1
  br label %523, !llvm.loop !20

539:                                              ; preds = %523
  %540 = shl i32 %521, 1
  %541 = trunc i32 %540 to i16
  br label %520, !llvm.loop !21

542:                                              ; preds = %552, %520
  %.0728 = phi i16 [ %560, %552 ], [ 1, %520 ]
  %.1725 = phi <4 x float> [ %558, %552 ], [ %.0724, %520 ]
  %.0713 = phi <4 x float> [ %.1, %552 ], [ undef, %520 ]
  %543 = zext i16 %.0728 to i32
  %544 = icmp slt i32 %543, 8
  br i1 %544, label %545, label %561

545:                                              ; preds = %547, %542
  %.0714 = phi i32 [ %551, %547 ], [ 0, %542 ]
  %.1 = phi <4 x float> [ %550, %547 ], [ %.0713, %542 ]
  %546 = icmp slt i32 %.0714, 4
  br i1 %546, label %547, label %552

547:                                              ; preds = %545
  %548 = extractelement <4 x float> %.1725, i32 %.0714
  %549 = call spir_func float @_Z21sub_group_shuffle_xorfj(float noundef %548, i32 noundef %543) #6
  %550 = insertelement <4 x float> %.1, float %549, i32 %.0714
  %551 = add nsw i32 %.0714, 1
  br label %545, !llvm.loop !22

552:                                              ; preds = %545
  %553 = zext i16 %85 to i32
  %554 = and i32 %553, %543
  %555 = icmp ne i32 %554, 0
  %556 = fsub <4 x float> %.1, %.1725
  %557 = fadd <4 x float> %.1, %.1725
  %558 = select i1 %555, <4 x float> %556, <4 x float> %557
  %559 = shl i32 %543, 1
  %560 = trunc i32 %559 to i16
  br label %542, !llvm.loop !23

561:                                              ; preds = %542
  %562 = call spir_func float @_Z4sqrtf(float noundef 3.200000e+01) #5, !fpmath !24
  %563 = insertelement <4 x float> poison, float %562, i64 0
  %564 = shufflevector <4 x float> %563, <4 x float> poison, <4 x i32> zeroinitializer
  %565 = fdiv <4 x float> %.1725, %564, !fpmath !17
  br label %566

566:                                              ; preds = %561, %512
  %.1747 = phi <4 x float> [ %565, %561 ], [ %.0746, %512 ]
  %567 = add i32 %.0.copyload, -1
  %568 = icmp ult i32 %567, 2
  br i1 %568, label %613, label %569

569:                                              ; preds = %566
  %570 = load i32, ptr addrspace(2) %7, align 4
  %571 = mul i32 %52, %570
  %572 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %2, i32 %571
  br i1 %93, label %573, label %598

573:                                              ; preds = %569
  %574 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %572, i32 %84
  %575 = addrspacecast ptr %28 to ptr addrspace(4)
  br label %576

576:                                              ; preds = %578, %573
  %.0722 = phi i32 [ 0, %573 ], [ %588, %578 ]
  %577 = icmp slt i32 %.0722, 4
  br i1 %577, label %578, label %589

578:                                              ; preds = %576
  %579 = extractelement <4 x float> %.1747, i32 %.0722
  %580 = bitcast float %579 to i32
  %581 = add i32 %580, 32767
  %582 = lshr i32 %580, 16
  %583 = and i32 %582, 1
  %584 = add i32 %581, %583
  %585 = lshr i32 %584, 16
  %586 = trunc i32 %585 to i16
  %587 = getelementptr inbounds [4 x i16], ptr addrspace(4) %575, i32 0, i32 %.0722
  store i16 %586, ptr addrspace(4) %587, align 2
  %588 = add nsw i32 %.0722, 1
  br label %576, !llvm.loop !16

589:                                              ; preds = %576
  %590 = addrspacecast ptr addrspace(1) %574 to ptr addrspace(4)
  br label %591

591:                                              ; preds = %593, %589
  %.0723 = phi i32 [ 0, %589 ], [ %597, %593 ]
  %592 = icmp slt i32 %.0723, 4
  br i1 %592, label %593, label %_Z11GatedActMulIN5metal6bfloatEEvPU3AS1KT_S4_PU3AS1S2_PU3AS1cPU3AS1fPU3AS1iPU3AS1KiRU3AS2KjSG_SG_SG_RU3AS2KN3uzu15activation_type14ActivationTypeERU3AS2KfSN_SN_SN_SN_NSH_13gated_act_mul13GatedActMulOpEbbbjjbbbjjj13ThreadContext.exit

593:                                              ; preds = %591
  %594 = getelementptr inbounds [4 x i16], ptr addrspace(4) %575, i32 0, i32 %.0723
  %595 = load i16, ptr addrspace(4) %594, align 2
  %596 = getelementptr inbounds [4 x i16], ptr addrspace(4) %590, i32 0, i32 %.0723
  store i16 %595, ptr addrspace(4) %596, align 2
  %597 = add nsw i32 %.0723, 1
  br label %591, !llvm.loop !13

598:                                              ; preds = %600, %569
  %.0751 = phi i32 [ %612, %600 ], [ 0, %569 ]
  %599 = icmp ult i32 %.0751, %92
  br i1 %599, label %600, label %_Z11GatedActMulIN5metal6bfloatEEvPU3AS1KT_S4_PU3AS1S2_PU3AS1cPU3AS1fPU3AS1iPU3AS1KiRU3AS2KjSG_SG_SG_RU3AS2KN3uzu15activation_type14ActivationTypeERU3AS2KfSN_SN_SN_SN_NSH_13gated_act_mul13GatedActMulOpEbbbjjbbbjjj13ThreadContext.exit

600:                                              ; preds = %598
  %601 = extractelement <4 x float> %.1747, i32 %.0751
  %602 = bitcast float %601 to i32
  %603 = add i32 %602, 32767
  %604 = lshr i32 %602, 16
  %605 = and i32 %604, 1
  %606 = add i32 %603, %605
  %607 = lshr i32 %606, 16
  %608 = trunc i32 %607 to i16
  %609 = add i32 %84, %.0751
  %610 = getelementptr inbounds nuw %"struct.metal::bfloat", ptr addrspace(1) %572, i32 %609
  %611 = addrspacecast ptr addrspace(1) %610 to ptr addrspace(4)
  store i16 %608, ptr addrspace(4) %611, align 2
  %612 = add i32 %.0751, 1
  br label %598, !llvm.loop !25

613:                                              ; preds = %566
  %614 = icmp ugt i32 %92, 0
  %615 = load i32, ptr addrspace(2) %7, align 4
  %616 = icmp eq i32 %.0.copyload, 2
  %617 = zext i1 %614 to i8
  %618 = zext i1 %616 to i8
  %619 = extractelement <4 x float> %.1747, i32 0
  %620 = call spir_func float @_Z4fabsf(float noundef %619) #5
  %621 = extractelement <4 x float> %.1747, i32 1
  %622 = call spir_func float @_Z4fabsf(float noundef %621) #5
  %623 = call spir_func float @_Z3maxff(float noundef %620, float noundef %622) #5
  %624 = extractelement <4 x float> %.1747, i32 2
  %625 = call spir_func float @_Z4fabsf(float noundef %624) #5
  %626 = extractelement <4 x float> %.1747, i32 3
  %627 = call spir_func float @_Z4fabsf(float noundef %626) #5
  %628 = call spir_func float @_Z3maxff(float noundef %625, float noundef %627) #5
  %629 = call spir_func float @_Z3maxff(float noundef %623, float noundef %628) #5
  %630 = udiv i32 %21, 4
  br label %631

631:                                              ; preds = %634, %613
  %.0712 = phi i16 [ 1, %613 ], [ %638, %634 ]
  %.0711 = phi float [ %629, %613 ], [ %636, %634 ]
  %632 = zext i16 %.0712 to i32
  %633 = icmp ult i32 %632, %630
  br i1 %633, label %634, label %639

634:                                              ; preds = %631
  %635 = call spir_func float @_Z21sub_group_shuffle_xorfj(float noundef %.0711, i32 noundef %632) #6
  %636 = call spir_func float @_Z3maxff(float noundef %.0711, float noundef %635) #5
  %637 = shl i32 %632, 1
  %638 = trunc i32 %637 to i16
  br label %631, !llvm.loop !26

639:                                              ; preds = %631
  %640 = call spir_func i32 @_Z8isfinitef(float noundef %.0711) #5
  %641 = icmp ne i32 %640, 0
  %642 = fcmp ogt float %.0711, 0.000000e+00
  %or.cond875 = select i1 %641, i1 %642, i1 false
  %643 = fdiv float %.0711, 1.270000e+02, !fpmath !17
  %644 = select i1 %or.cond875, float %643, float 1.000000e+00
  br label %645

645:                                              ; preds = %647, %639
  %.0721 = phi i32 [ 0, %639 ], [ %661, %647 ]
  %.0720 = phi i32 [ 0, %639 ], [ %660, %647 ]
  %.0719 = phi i32 [ 0, %639 ], [ %655, %647 ]
  %646 = icmp ult i32 %.0721, 4
  br i1 %646, label %647, label %662

647:                                              ; preds = %645
  %648 = extractelement <4 x float> %.1747, i32 %.0721
  %649 = fdiv float %648, %644, !fpmath !17
  %650 = call spir_func float @_Z5roundf(float noundef %649) #5
  %651 = call spir_func float @_Z5clampfff(float noundef %650, float noundef -1.270000e+02, float noundef 1.270000e+02) #5
  %652 = fptosi float %651 to i8
  %653 = getelementptr inbounds nuw [4 x i8], ptr %27, i32 0, i32 %.0721
  store i8 %652, ptr %653, align 1
  %654 = sext i8 %652 to i32
  %655 = add nsw i32 %.0719, %654
  %656 = zext i8 %652 to i32
  %657 = mul i32 8, %.0721
  %658 = and i32 %657, 31
  %659 = shl i32 %656, %658
  %660 = or i32 %.0720, %659
  %661 = add i32 %.0721, 1
  br label %645, !llvm.loop !27

662:                                              ; preds = %645
  %663 = mul i32 %52, %615
  br i1 %39, label %664, label %692

664:                                              ; preds = %662
  %665 = call spir_func i32 @_Z21sub_group_shuffle_xorjj(i32 noundef %.0720, i32 noundef 1) #6
  %666 = zext i16 %85 to i32
  %667 = and i32 %666, 1
  %668 = icmp ne i32 %667, 0
  %..0720 = select i1 %668, i32 %665, i32 %.0720
  %669 = select i1 %668, i32 %.0720, i32 %665
  br i1 %614, label %670, label %.critedge

670:                                              ; preds = %664
  %671 = mul i32 8, %667
  %672 = and i32 %671, 31
  %673 = lshr i32 %..0720, %672
  %674 = trunc i32 %673 to i8
  %675 = add i32 %671, 16
  %676 = and i32 %675, 31
  %677 = lshr i32 %..0720, %676
  %678 = trunc i32 %677 to i8
  %679 = lshr i32 %669, %672
  %680 = trunc i32 %679 to i8
  %681 = lshr i32 %669, %676
  %682 = trunc i32 %681 to i8
  %683 = insertelement <4 x i8> poison, i8 %674, i32 0
  %684 = insertelement <4 x i8> %683, i8 %678, i32 1
  %685 = insertelement <4 x i8> %684, i8 %680, i32 2
  %686 = insertelement <4 x i8> %685, i8 %682, i32 3
  %687 = getelementptr inbounds nuw i8, ptr addrspace(1) %3, i32 %663
  %688 = and i32 %84, -8
  %689 = getelementptr inbounds nuw i8, ptr addrspace(1) %687, i32 %688
  %690 = mul i32 4, %667
  %691 = getelementptr inbounds nuw i8, ptr addrspace(1) %689, i32 %690
  store <4 x i8> %686, ptr addrspace(1) %691, align 4
  br label %707

692:                                              ; preds = %662
  br i1 %614, label %693, label %.critedge

693:                                              ; preds = %692
  %694 = load i8, ptr %27, align 1
  %695 = getelementptr inbounds [4 x i8], ptr %27, i32 0, i32 1
  %696 = load i8, ptr %695, align 1
  %697 = getelementptr inbounds [4 x i8], ptr %27, i32 0, i32 2
  %698 = load i8, ptr %697, align 1
  %699 = getelementptr inbounds [4 x i8], ptr %27, i32 0, i32 3
  %700 = load i8, ptr %699, align 1
  %701 = insertelement <4 x i8> poison, i8 %694, i32 0
  %702 = insertelement <4 x i8> %701, i8 %696, i32 1
  %703 = insertelement <4 x i8> %702, i8 %698, i32 2
  %704 = insertelement <4 x i8> %703, i8 %700, i32 3
  %705 = getelementptr inbounds nuw i8, ptr addrspace(1) %3, i32 %663
  %706 = getelementptr inbounds nuw i8, ptr addrspace(1) %705, i32 %84
  store <4 x i8> %704, ptr addrspace(1) %706, align 4
  br label %707

707:                                              ; preds = %693, %670
  br i1 %614, label %708, label %.critedge

708:                                              ; preds = %707
  %709 = zext i16 %85 to i32
  %710 = urem i32 %709, %630
  %711 = icmp eq i32 %710, 0
  br i1 %711, label %712, label %.critedge

712:                                              ; preds = %708
  %713 = udiv i32 %615, %21
  %714 = mul i32 %52, %713
  %715 = udiv i32 %84, %21
  %716 = add i32 %714, %715
  %717 = getelementptr inbounds nuw float, ptr addrspace(1) %4, i32 %716
  store float %644, ptr addrspace(1) %717, align 4
  br label %.critedge

.critedge:                                        ; preds = %712, %708, %707, %692, %664
  br i1 %616, label %718, label %_Z11GatedActMulIN5metal6bfloatEEvPU3AS1KT_S4_PU3AS1S2_PU3AS1cPU3AS1fPU3AS1iPU3AS1KiRU3AS2KjSG_SG_SG_RU3AS2KN3uzu15activation_type14ActivationTypeERU3AS2KfSN_SN_SN_SN_NSH_13gated_act_mul13GatedActMulOpEbbbjjbbbjjj13ThreadContext.exit

718:                                              ; preds = %.critedge
  %719 = udiv i32 %22, 4
  br label %720

720:                                              ; preds = %723, %718
  %.0710 = phi i16 [ 1, %718 ], [ %727, %723 ]
  %.0709 = phi i32 [ %.0719, %718 ], [ %725, %723 ]
  %721 = zext i16 %.0710 to i32
  %722 = icmp ult i32 %721, %719
  br i1 %722, label %723, label %728

723:                                              ; preds = %720
  %724 = call spir_func i32 @_Z21sub_group_shuffle_xorij(i32 noundef %.0709, i32 noundef %721) #6
  %725 = add nsw i32 %.0709, %724
  %726 = shl i32 %721, 1
  %727 = trunc i32 %726 to i16
  br label %720, !llvm.loop !28

728:                                              ; preds = %720
  br i1 %614, label %729, label %_Z11GatedActMulIN5metal6bfloatEEvPU3AS1KT_S4_PU3AS1S2_PU3AS1cPU3AS1fPU3AS1iPU3AS1KiRU3AS2KjSG_SG_SG_RU3AS2KN3uzu15activation_type14ActivationTypeERU3AS2KfSN_SN_SN_SN_NSH_13gated_act_mul13GatedActMulOpEbbbjjbbbjjj13ThreadContext.exit

729:                                              ; preds = %728
  %730 = zext i16 %85 to i32
  %731 = urem i32 %730, %719
  %732 = icmp eq i32 %731, 0
  br i1 %732, label %733, label %_Z11GatedActMulIN5metal6bfloatEEvPU3AS1KT_S4_PU3AS1S2_PU3AS1cPU3AS1fPU3AS1iPU3AS1KiRU3AS2KjSG_SG_SG_RU3AS2KN3uzu15activation_type14ActivationTypeERU3AS2KfSN_SN_SN_SN_NSH_13gated_act_mul13GatedActMulOpEbbbjjbbbjjj13ThreadContext.exit

733:                                              ; preds = %729
  %734 = udiv i32 %615, %22
  %735 = mul i32 %52, %734
  %736 = udiv i32 %84, %22
  %737 = add i32 %735, %736
  %738 = getelementptr inbounds nuw i32, ptr addrspace(1) %5, i32 %737
  store i32 %.0709, ptr addrspace(1) %738, align 4
  br label %_Z11GatedActMulIN5metal6bfloatEEvPU3AS1KT_S4_PU3AS1S2_PU3AS1cPU3AS1fPU3AS1iPU3AS1KiRU3AS2KjSG_SG_SG_RU3AS2KN3uzu15activation_type14ActivationTypeERU3AS2KfSN_SN_SN_SN_NSH_13gated_act_mul13GatedActMulOpEbbbjjbbbjjj13ThreadContext.exit

_Z11GatedActMulIN5metal6bfloatEEvPU3AS1KT_S4_PU3AS1S2_PU3AS1cPU3AS1fPU3AS1iPU3AS1KiRU3AS2KjSG_SG_SG_RU3AS2KN3uzu15activation_type14ActivationTypeERU3AS2KfSN_SN_SN_SN_NSH_13gated_act_mul13GatedActMulOpEbbbjjbbbjjj13ThreadContext.exit: ; preds = %733, %729, %728, %.critedge, %598, %591
  call void @llvm.lifetime.end.p0(ptr %27)
  call void @llvm.lifetime.end.p0(ptr %28)
  call void @llvm.lifetime.end.p0(ptr %.sroa.17)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5798)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5801)
  call void @llvm.lifetime.end.p0(ptr %.sroa.7844)
  call void @llvm.lifetime.end.p0(ptr %29)
  call void @llvm.lifetime.end.p0(ptr %30)
  call void @llvm.lifetime.end.p0(ptr %31)
  call void @llvm.lifetime.end.p0(ptr %32)
  call void @llvm.lifetime.end.p0(ptr %33)
  call void @llvm.lifetime.end.p0(ptr %34)
  call void @llvm.lifetime.end.p0(ptr %35)
  call void @llvm.lifetime.end.p0(ptr %36)
  call void @llvm.lifetime.end.p0(ptr %37)
  call void @llvm.lifetime.end.p0(ptr %38)
  call void @llvm.lifetime.end.p0(ptr %.sroa.6839)
  call void @llvm.lifetime.end.p0(ptr %.sroa.5836)
  call void @llvm.lifetime.end.p0(ptr %.sroa.6833)
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z12get_group_idj(i32 noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z12get_local_idj(i32 noundef) #1

; Function Attrs: convergent nounwind
declare dso_local spir_func i32 @_Z22get_sub_group_local_idv() #2

; Function Attrs: convergent nounwind
declare dso_local spir_func i32 @_Z16get_sub_group_idv() #2

; Function Attrs: convergent nounwind
declare dso_local spir_func i32 @_Z18get_sub_group_sizev() #2

; Function Attrs: convergent nounwind
declare dso_local spir_func i32 @_Z18get_num_sub_groupsv() #2

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z14get_local_sizej(i32 noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z14get_num_groupsj(i32 noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z15get_global_sizej(i32 noundef) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p4.i32(ptr noalias writeonly captures(none), ptr addrspace(4) noalias readonly captures(none), i32, i1 immarg) #3

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z3minjj(i32 noundef, i32 noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func <4 x float> @_Z5clampDv4_fff(<4 x float> noundef, float noundef, float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z4fabsf(float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z10native_expf(float noundef) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #4

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z4tanhf(float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z3fmafff(float noundef, float noundef, float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z3expf(float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z8copysignff(float noundef, float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z10native_logf(float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z4sqrtf(float noundef) #1

; Function Attrs: convergent nounwind
declare dso_local spir_func float @_Z21sub_group_shuffle_xorfj(float noundef, i32 noundef) #2

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z3maxff(float noundef, float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z8isfinitef(float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z5clampfff(float noundef, float noundef, float noundef) #1

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func float @_Z5roundf(float noundef) #1

; Function Attrs: convergent nounwind
declare dso_local spir_func i32 @_Z21sub_group_shuffle_xorjj(i32 noundef, i32 noundef) #2

; Function Attrs: convergent nounwind
declare dso_local spir_func i32 @_Z21sub_group_shuffle_xorij(i32 noundef, i32 noundef) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; uselistorder directives
uselistorder ptr @_Z12get_group_idj, { 2, 1, 0 }
uselistorder ptr @_Z14get_local_sizej, { 2, 1, 0 }
uselistorder ptr @_Z14get_num_groupsj, { 2, 1, 0 }
uselistorder ptr @_Z15get_global_sizej, { 2, 1, 0 }
uselistorder ptr @llvm.memcpy.p0.p4.i32, { 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @_Z5clampDv4_fff, { 1, 0 }
uselistorder ptr @_Z4fabsf, { 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @_Z10native_expf, { 2, 1, 0 }
uselistorder ptr @_Z3fmafff, { 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @_Z21sub_group_shuffle_xorfj, { 1, 0 }
uselistorder ptr @_Z3maxff, { 3, 2, 1, 0 }
uselistorder ptr @llvm.memcpy.p0.p0.i64, { 8, 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.lifetime.start.p0, { 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.lifetime.end.p0, { 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
attributes #1 = { convergent nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind willreturn memory(none) "uniform-work-group-size" }
attributes #6 = { convergent nounwind "uniform-work-group-size" }

!opencl.ocl.version = !{!0}
!opencl.cxx.version = !{!1}
!opencl.spir.version = !{!0}
!llvm.module.flags = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 3, i32 0}
!1 = !{i32 2021, i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 23.1.1"}
!4 = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!6 = !{!"bfloat*", !"bfloat*", !"bfloat*", !"int8_t*", !"float*", !"int32_t*", !"int32_t*", !"uint*", !"uint*", !"uint*", !"uint*", !"ActivationType*", !"float*", !"float*", !"float*", !"float*", !"float*", !"GatedActMulOp*", !"uint", !"uint", !"uint", !"uint", !"uint", !"uint", !"uint", !"uint"}
!7 = !{!"metal::bfloat*", !"metal::bfloat*", !"metal::bfloat*", !"char*", !"float*", !"int*", !"int*", !"uint*", !"uint*", !"uint*", !"uint*", !"uzu::activation_type::ActivationType*", !"float*", !"float*", !"float*", !"float*", !"float*", !"uzu::gated_act_mul::GatedActMulOp*", !"uint", !"uint", !"uint", !"uint", !"uint", !"uint", !"uint", !"uint"}
!8 = !{!"const", !"const", !"", !"", !"", !"", !"const", !"const", !"const", !"const", !"const", !"const", !"const", !"const", !"const", !"const", !"const", !"const", !"", !"", !"", !"", !"", !"", !"", !""}
!9 = !{!"act_operand", !"value_operand", !"fp_out", !"q_out", !"scales_out", !"group_sums_out", !"hadamard_factors", !"gated_dim_m2v_p", !"batch_dim_m2v_p", !"value_offset_m2v_p", !"value_row_stride_m2v_p", !"act_type_m2v_p", !"activation_alpha_m2v_p", !"gate_clip_min_m2v_p", !"gate_clip_max_m2v_p", !"value_clip_min_m2v_p", !"value_clip_max_m2v_p", !"ops_m2v_p", !"grouped_by_weight_nibble_m2v_u", !"interleaved_m2v_u", !"use_hadamard_m2v_u", !"activation_scale_group_size", !"sum_group_size", !"custom_activation_alpha_m2v_u", !"clip_gate_m2v_u", !"clip_value_m2v_u"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
!13 = distinct !{!13, !11}
!14 = distinct !{!14, !11}
!15 = distinct !{!15, !11}
!16 = distinct !{!16, !11}
!17 = !{float 2.500000e+00}
!18 = distinct !{!18, !11, !19}
!19 = !{!"llvm.loop.unroll.full"}
!20 = distinct !{!20, !11, !19}
!21 = distinct !{!21, !11}
!22 = distinct !{!22, !11}
!23 = distinct !{!23, !11}
!24 = !{float 3.000000e+00}
!25 = distinct !{!25, !11}
!26 = distinct !{!26, !11}
!27 = distinct !{!27, !11, !19}
!28 = distinct !{!28, !11}
