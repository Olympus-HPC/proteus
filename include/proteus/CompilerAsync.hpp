//===-- CompilerAsync.hpp -- Asynchronous Compiler header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Asynchronous Compiler that internally uses the AsyncBuilder
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_ASYNC_COMPILER_HPP
#define PROTEUS_ASYNC_COMPILER_HPP

#include <llvm/Support/MemoryBuffer.h>

#include "proteus/AsyncBuilder.hpp"
#include "proteus/CompilationTask.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

/**
 * @brief Asynchronous compiler that internally uses AsyncBuilder
 * 
 * This class provides backward compatibility with the existing codebase.
 * It's a bridge between the old asynchronous API and the new Builder architecture.
 */
class CompilerAsync {
public:
  /**
   * @brief Get the singleton instance of CompilerAsync
   * 
   * @param NumThreads Number of worker threads to use
   */
  static CompilerAsync& instance(int NumThreads) {
    static CompilerAsync Singleton{NumThreads};
    return Singleton;
  }

  /**
   * @brief Queue a task for asynchronous compilation
   * 
   * @param CT The task to compile
   */
  void compile(CompilationTask&& CT) {
    HashT HashValue = CT.getHashValue();
    
    // Use the AsyncBuilder to queue the task
    AsyncBuilder::instance(NumThreads).build(CT);
  }

  /**
   * @brief Join all worker threads
   */
  void joinAllThreads() {
    AsyncBuilder::instance(NumThreads).joinAllThreads();
  }

  /**
   * @brief Check if a compilation is pending
   * 
   * @param HashValue Hash of the compilation task
   * @return True if the compilation is pending
   */
  bool isCompilationPending(const HashT& HashValue) {
    return AsyncBuilder::instance(NumThreads).isCompilationPending(HashValue);
  }

  /**
   * @brief Get a compilation result, optionally waiting for it to complete
   * 
   * @param HashValue Hash of the compilation task
   * @param BlockingWait Whether to wait for the result to be ready
   * @return The memory buffer with the compiled object, or nullptr if not ready and not waiting
   */
  std::unique_ptr<MemoryBuffer> takeCompilationResult(const HashT& HashValue, bool BlockingWait) {
    // Get the result from the AsyncBuilder
    std::unique_ptr<CompilationResult> Result = 
        AsyncBuilder::instance(NumThreads).getResult(HashValue, BlockingWait);
    
    // Return only the memory buffer for backward compatibility
    return Result ? Result->takeObjectBuffer() : nullptr;
  }

private:
  int NumThreads;

  // Private constructor for singleton
  explicit CompilerAsync(int NumThreads) : NumThreads(NumThreads) {}
  ~CompilerAsync() = default;
};

} // namespace proteus

#endif // PROTEUS_ASYNC_COMPILER_HPP