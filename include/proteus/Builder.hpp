//===-- Builder.hpp -- Compilation builder header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder transforms a CompilationTask into a CompilationResult.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_BUILDER_HPP
#define PROTEUS_BUILDER_HPP

#include <memory>

#include "proteus/CompilationTask.hpp"
#include "proteus/CompilationResult.hpp"

namespace proteus {

/**
 * @brief Abstract Builder interface for transforming CompilationTasks into CompilationResults
 * 
 * The Builder is responsible for taking a CompilationTask and producing a
 * CompilationResult, which contains the compiled function and associated metadata.
 * This abstraction allows for different compilation strategies and backends.
 */
class Builder {
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~Builder() = default;
  
  /**
   * @brief Build a CompilationResult from a CompilationTask
   * 
   * @param Task The task containing all information needed for compilation
   * @return A CompilationResult containing the compiled function and metadata
   */
  virtual std::unique_ptr<CompilationResult> build(const CompilationTask& Task) = 0;
};

} // namespace proteus

#endif // PROTEUS_BUILDER_HPP