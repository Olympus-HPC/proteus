//===-- CompilerSync.hpp -- Synchronous Compiler header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Synchronous Compiler that internally uses the SyncBuilder
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_SYNC_COMPILER_HPP
#define PROTEUS_SYNC_COMPILER_HPP

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/MemoryBuffer.h>

#include "proteus/CompilationTask.hpp"
#include "proteus/SyncBuilder.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

/**
 * @brief Synchronous compiler that internally uses SyncBuilder
 * 
 * This class provides backward compatibility with the existing codebase.
 * It's a bridge between the old compile() method that returns a MemoryBuffer
 * and the new Builder architecture that returns a CompilationResult.
 */
class CompilerSync {
public:
  /**
   * @brief Get the singleton instance of CompilerSync
   */
  static CompilerSync& instance() {
    static CompilerSync Singleton;
    return Singleton;
  }

  /**
   * @brief Compile a task and return the memory buffer
   * 
   * This method provides backward compatibility with the existing codebase.
   * 
   * @param CT The task to compile
   * @return The compiled object code in a memory buffer
   */
  std::unique_ptr<MemoryBuffer> compile(CompilationTask&& CT) {
    // Use the SyncBuilder to compile the task
    std::unique_ptr<CompilationResult> Result = SyncBuilder::instance().build(CT);
    
    // Return only the memory buffer for backward compatibility
    return Result->takeObjectBuffer();
  }

private:
  // Private constructor for singleton
  CompilerSync() = default;
  ~CompilerSync() = default;
};

} // namespace proteus

#endif // PROTEUS_SYNC_COMPILER_HPP