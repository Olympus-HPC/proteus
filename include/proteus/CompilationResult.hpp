//===-- CompilationResult.hpp -- Compilation Result header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represents the result of a successful compilation, containing the function
// pointer and associated metadata needed to execute a JIT-compiled function.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_COMPILATION_RESULT_HPP
#define PROTEUS_COMPILATION_RESULT_HPP

#include <llvm/Support/MemoryBuffer.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

/**
 * @brief Represents the result of a successful compilation.
 * 
 * Contains the function pointer and all associated metadata needed to execute
 * a JIT-compiled function.
 */
class CompilationResult {
private:
  HashT HashValue;
  std::string MangledName;
  std::unique_ptr<MemoryBuffer> ObjBuffer;
  void* FunctionPtr;
  SmallVector<RuntimeConstant> RuntimeConstants;

public:
  /**
   * @brief Construct a CompilationResult
   * 
   * @param HashValue Hash that uniquely identifies this compilation
   * @param MangledName Mangled name of the compiled function
   * @param ObjBuffer Memory buffer containing the compiled object code
   * @param FunctionPtr Pointer to the JIT-compiled function
   * @param RuntimeConstants Runtime constants used in this compilation
   */
  CompilationResult(
      HashT HashValue,
      std::string MangledName,
      std::unique_ptr<MemoryBuffer> ObjBuffer,
      void* FunctionPtr,
      const SmallVector<RuntimeConstant>& RuntimeConstants)
      : HashValue(HashValue),
        MangledName(std::move(MangledName)),
        ObjBuffer(std::move(ObjBuffer)),
        FunctionPtr(FunctionPtr),
        RuntimeConstants(RuntimeConstants) {}

  /**
   * @brief Get the hash value that uniquely identifies this compilation
   */
  HashT getHashValue() const { return HashValue; }

  /**
   * @brief Get the mangled name of the compiled function
   */
  const std::string& getMangledName() const { return MangledName; }

  /**
   * @brief Get the memory buffer containing the compiled object code
   */
  const MemoryBuffer& getObjectBuffer() const { return *ObjBuffer; }

  /**
   * @brief Take ownership of the memory buffer containing the compiled object code
   */
  std::unique_ptr<MemoryBuffer> takeObjectBuffer() { return std::move(ObjBuffer); }

  /**
   * @brief Get the function pointer for direct invocation
   * 
   * @tparam FuncType Function pointer type to cast to
   * @return The function pointer cast to the requested type
   */
  template<typename FuncType>
  FuncType getFunction() const {
    return reinterpret_cast<FuncType>(FunctionPtr);
  }

  /**
   * @brief Get the raw function pointer
   */
  void* getFunctionPtr() const { return FunctionPtr; }

  /**
   * @brief Get the runtime constants used in this compilation
   */
  const SmallVector<RuntimeConstant>& getRuntimeConstants() const { 
    return RuntimeConstants; 
  }
};

} // namespace proteus

#endif // PROTEUS_COMPILATION_RESULT_HPP