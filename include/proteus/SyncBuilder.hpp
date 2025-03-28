//===-- SyncBuilder.hpp -- Synchronous Builder header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SyncBuilder implements the Builder interface for synchronous compilation
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_SYNC_BUILDER_HPP
#define PROTEUS_SYNC_BUILDER_HPP

#include <memory>

#include "proteus/Builder.hpp"
#include "proteus/CompilationTask.hpp"
#include "proteus/CompilationResult.hpp"

namespace proteus {

/**
 * @brief Synchronous Builder implementation
 * 
 * The SyncBuilder executes compilation tasks synchronously in the current thread.
 */
class SyncBuilder : public Builder {
public:
  /**
   * @brief Get the singleton instance of SyncBuilder
   */
  static SyncBuilder& instance() {
    static SyncBuilder Singleton;
    return Singleton;
  }

  /**
   * @brief Build a CompilationResult from a CompilationTask
   * 
   * Executes the compilation synchronously in the current thread.
   * 
   * @param Task The task containing all information needed for compilation
   * @return A CompilationResult containing the compiled function and metadata
   */
  std::unique_ptr<CompilationResult> build(const CompilationTask& Task) override {
    // For now, we'll create a mock result with an empty memory buffer
    // In a real implementation, we would need to implement the compilation logic here,
    // since we can't copy or move from the const reference.
    
    // Create a simple memory buffer for now
    std::unique_ptr<MemoryBuffer> ObjBuffer = 
        MemoryBuffer::getMemBuffer("", "sync-builder-buffer");
    
    // Create and return a CompilationResult
    return std::make_unique<CompilationResult>(
        Task.getHashValue(),
        Task.getKernelName() + Task.getSuffix(),
        std::move(ObjBuffer),
        nullptr, // Function pointer would come from the compiled object
        Task.getRCValues());
  }

private:
  // Private constructor for singleton
  SyncBuilder() = default;
};

} // namespace proteus

#endif // PROTEUS_SYNC_BUILDER_HPP