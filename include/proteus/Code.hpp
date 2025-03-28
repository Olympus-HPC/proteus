//===-- Code.hpp -- Extracted code representation header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represents extracted function code in a format suitable for compilation.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_CODE_HPP
#define PROTEUS_CODE_HPP

#include <llvm/IR/Module.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/raw_ostream.h>

#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

/**
 * @brief Extracted code representation used for compilation.
 *
 * This class represents the extracted code (function or kernel) that will be
 * used for compilation. In the current implementation, this is LLVM IR.
 */
class Code {
private:
  std::unique_ptr<Module> Module;
  std::string FunctionName;
  bool Optimized;

public:
  /**
   * @brief Construct a Code object from an existing Module
   *
   * @param Module LLVM module containing the function to compile
   * @param FunctionName Name of the function within the module
   */
  Code(std::unique_ptr<llvm::Module> Module, std::string FunctionName)
      : Module(std::move(Module)), FunctionName(std::move(FunctionName)), Optimized(false) {}

  /**
   * @brief Copy constructor is deleted
   */
  Code(const Code&) = delete;

  /**
   * @brief Move constructor
   */
  Code(Code&& Other) noexcept
      : Module(std::move(Other.Module)),
        FunctionName(std::move(Other.FunctionName)),
        Optimized(Other.Optimized) {}

  /**
   * @brief Copy assignment is deleted
   */
  Code& operator=(const Code&) = delete;

  /**
   * @brief Move assignment
   */
  Code& operator=(Code&& Other) noexcept {
    if (this != &Other) {
      Module = std::move(Other.Module);
      FunctionName = std::move(Other.FunctionName);
      Optimized = Other.Optimized;
    }
    return *this;
  }

  /**
   * @brief Get the function name
   */
  const std::string& getFunctionName() const { return FunctionName; }

  /**
   * @brief Get a reference to the underlying module
   */
  llvm::Module& getModule() { return *Module; }

  /**
   * @brief Get a const reference to the underlying module
   */
  const llvm::Module& getModule() const { return *Module; }

  /**
   * @brief Take ownership of the underlying module
   */
  std::unique_ptr<llvm::Module> takeModule() { return std::move(Module); }

  /**
   * @brief Create a clone of this Code object with its own module copy
   */
  std::unique_ptr<Code> clone(LLVMContext& Ctx) const {
    SmallVector<char, 4096> ModuleStr;
    raw_svector_ostream OS(ModuleStr);
    WriteBitcodeToFile(*Module, OS);
    StringRef ModuleStrRef = StringRef{ModuleStr.data(), ModuleStr.size()};
    auto BufferRef = MemoryBufferRef{ModuleStrRef, ""};
    auto ClonedModule = parseBitcodeFile(BufferRef, Ctx);
    if (auto E = ClonedModule.takeError()) {
      PROTEUS_FATAL_ERROR("Failed to parse bitcode" + toString(std::move(E)));
    }

    return std::make_unique<Code>(std::move(ClonedModule.get()), FunctionName);
  }

  /**
   * @brief Mark the code as optimized
   */
  void markOptimized() { Optimized = true; }

  /**
   * @brief Check if the code has been optimized
   */
  bool isOptimized() const { return Optimized; }

  /**
   * @brief Compute a hash value for this code
   */
  HashT getHash() const {
    // This is a simple implementation - in a full implementation we would
    // want to hash the actual LLVM IR content
    return hash(FunctionName);
  }
};

} // namespace proteus

#endif // PROTEUS_CODE_HPP