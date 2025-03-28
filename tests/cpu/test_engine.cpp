//===-- test_engine.cpp -- Test for Engine component --===//
//
// Part of Proteus Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests the new Engine component architecture.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <iostream>
#include <vector>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include "proteus/Code.hpp"
#include "proteus/Engine.hpp"
#include "proteus/Utils.h"

using namespace proteus;
using namespace llvm;

// Host function to compute daxpy: y = alpha*x + y
void daxpy(float alpha, float* x, float* y, int n) {
  for (int i = 0; i < n; i++) {
    y[i] = alpha * x[i] + y[i];
  }
}

// Simple IR module for testing
const char* DaxpyIR = R"(
define void @daxpy(float %alpha, float* %x, float* %y, i32 %n) {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds float, float* %x, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %mul = fmul float %0, %alpha
  %arrayidx3 = getelementptr inbounds float, float* %y, i64 %idxprom
  %1 = load float, float* %arrayidx3, align 4
  %add = fadd float %mul, %1
  store float %add, float* %arrayidx3, align 4
  %inc = add nuw nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
)";

// Add metadata to mark an argument as a runtime constant
void addRuntimeConstantMetadata(Module& M, const std::string& FnName, int ArgIdx) {
  Function* F = M.getFunction(FnName);
  if (!F) return;
  
  LLVMContext& Ctx = M.getContext();
  MDNode* ArgNode = MDNode::get(Ctx, {
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), ArgIdx))
  });
  F->setMetadata("jit_arg_nos", ArgNode);
}

// Test the Engine component with a simple daxpy kernel
int main() {
  // Create test data
  const int N = 1000;
  std::vector<float> x(N, 1.0f);
  std::vector<float> y(N, 2.0f);
  std::vector<float> expected(N);
  float alpha = 3.0f;
  
  // Compute expected results using host function
  daxpy(alpha, x.data(), y.data(), N);
  expected = y;
  
  // Reset y for JIT test
  std::fill(y.begin(), y.end(), 2.0f);
  
  try {
    // Create a CPU engine
    auto Engine = proteus::Engine::create(BackendType::CPU);
    
    // Configure the engine
    EngineConfig Config;
    Config.SpecializeArgs = true;
    Config.AsyncCompilation = false;
    Engine->setConfig(Config);
    
    // Parse the IR module
    LLVMContext Ctx;
    SMDiagnostic Err;
    auto M = parseIR(MemoryBufferRef(DaxpyIR, "daxpy_module"), Err, Ctx);
    if (!M) {
      std::cerr << "Error parsing IR\n";
      return 1;
    }
    
    // Add metadata for runtime constant specialization (alpha as constant)
    addRuntimeConstantMetadata(*M, "daxpy", 0);
    
    // Create Code object
    auto DaxpyCode = std::make_unique<Code>(std::move(M), "daxpy");
    
    // Create runtime constant for alpha
    SmallVector<RuntimeConstant> RCs;
    RuntimeConstant RC;
    RC.ArgIdx = 0;
    RC.Type = PROTEUS_F32;
    RC.Value.Float = alpha;
    RCs.push_back(RC);
    
    // Create a compilation task
    auto Task = Engine->createCompilationTask(*DaxpyCode, "daxpy", 
                                             dim3(1,1,1), dim3(1,1,1), RCs);
    
    // Compile the task
    auto Result = Engine->compile(*Task);
    if (!Result) {
      std::cerr << "Compilation failed\n";
      return 1;
    }
    
    // Get the function pointer
    auto JitFunc = Result->getFunction<void(*)(float*, float*, int)>();
    if (!JitFunc) {
      std::cerr << "Failed to get function pointer\n";
      return 1;
    }
    
    // Time the JIT function execution
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute the JIT function (note: alpha is specialized, so not passed)
    JitFunc(x.data(), y.data(), N);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
      if (std::abs(y[i] - expected[i]) > 1e-6) {
        std::cerr << "Mismatch at " << i << ": " << y[i] << " vs " << expected[i] << "\n";
        correct = false;
        break;
      }
    }
    
    if (correct) {
      std::cout << "Test PASSED. JIT execution time: " << elapsed.count() << " ms\n";
      return 0;
    } else {
      std::cerr << "Test FAILED.\n";
      return 1;
    }
    
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
}