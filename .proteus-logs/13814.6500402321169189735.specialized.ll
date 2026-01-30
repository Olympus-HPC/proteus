; ModuleID = 'JitModule'
source_filename = "/Users/beckingsale1/workspaces/proteus-yolo/proteus/tests/cpu/lambda_auto_readonly.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.7.0"

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define dso_local void @"_ZZ4mainENK3$_0clEv.jit.6500402321169189735"(ptr noundef nonnull align 8 dereferenceable(29) %0) local_unnamed_addr #0 !proteus.jit !7 !jit_arg_nos !8 {
  %2 = getelementptr inbounds i8, ptr %0, i64 8
  %3 = load i32, ptr %2, align 8
  %4 = sitofp i32 %3 to double
  %5 = load ptr, ptr %0, align 8
  store double %4, ptr %5, align 8
  %6 = getelementptr inbounds i8, ptr %0, i64 16
  %7 = load double, ptr %6, align 8
  %8 = load ptr, ptr %0, align 8
  %9 = getelementptr inbounds i8, ptr %8, i64 8
  store double %7, ptr %9, align 8
  %10 = getelementptr inbounds i8, ptr %0, i64 24
  %11 = load float, ptr %10, align 8
  %12 = fpext float %11 to double
  %13 = load ptr, ptr %0, align 8
  %14 = getelementptr inbounds i8, ptr %13, i64 16
  store double %12, ptr %14, align 8
  %15 = getelementptr inbounds i8, ptr %0, i64 28
  %16 = load i8, ptr %15, align 4
  %17 = trunc i8 %16 to i1
  %18 = select i1 %17, double 1.000000e+00, double 0.000000e+00
  %19 = load ptr, ptr %0, align 8
  %20 = getelementptr inbounds i8, ptr %19, i64 24
  store double %18, ptr %20, align 8
  ret void
}

attributes #0 = { mustprogress noinline nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+jsconv,+lse,+neon,+pauth,+ras,+rcpc,+rdm,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 1]}
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = !{!"proteus.jit"}
!8 = !{}
