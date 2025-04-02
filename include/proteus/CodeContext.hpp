//===-- CodeContext.hpp -- Code context and registry header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CodeContext maintains a registry of functions and lambdas available for JIT
// compilation, along with their metadata.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_CODE_CONTEXT_HPP
#define PROTEUS_CODE_CONTEXT_HPP

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
#include <unordered_map>
#include <optional>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Error.h"
#include "proteus/Logger.hpp"

namespace proteus {

using namespace llvm;

/**
 * @brief Information about a registered function
 */
struct FunctionInfo {
  void* FunctionPtr;
  SmallVector<int32_t> RuntimeConstantIndices;
  SmallVector<int32_t> RuntimeConstantTypes;
};

/**
 * @brief Information about a registered lambda
 */
struct LambdaInfo {
  StringRef LambdaType;
  SmallVector<RuntimeConstant> CaptureValues;
};

/**
 * @brief Registry for functions and lambdas available for JIT compilation
 * 
 * CodeContext maintains a registry of functions and lambdas available for JIT
 * compilation, along with their metadata such as runtime constant argument
 * indices, types, and capture values.
 */
class CodeContext {
public:
  /**
   * @brief Get the singleton instance of CodeContext
   */
  static CodeContext& instance() {
    static CodeContext Singleton;
    return Singleton;
  }

  /**
   * @brief Register a function for JIT compilation
   * 
   * @param Name Function name
   * @param FuncPtr Pointer to the function
   * @param RCIndices Indices of runtime constant arguments
   * @param RCTypes Types of runtime constant arguments
   * @param NumRCs Number of runtime constant arguments
   */
  void registerFunction(StringRef Name, void* FuncPtr, 
                       const int32_t* RCIndices, const int32_t* RCTypes, int32_t NumRCs) {
    FunctionInfo Info;
    Info.FunctionPtr = FuncPtr;
    Info.RuntimeConstantIndices.append(RCIndices, RCIndices + NumRCs);
    Info.RuntimeConstantTypes.append(RCTypes, RCTypes + NumRCs);
    FunctionMap[Name.str()] = std::move(Info);
  }

  /**
   * @brief Look up a registered function by name
   * 
   * @param Name Function name
   * @return Optional reference to function information, or nullopt if not found
   */
  std::optional<std::reference_wrapper<const FunctionInfo>> 
  lookupFunction(StringRef Name) const {
    auto It = FunctionMap.find(Name.str());
    if (It == FunctionMap.end()) {
      return std::nullopt;
    }
    return std::cref(It->second);
  }

  /**
   * @brief Push a JIT variable for the next lambda registration
   * 
   * @param RC Runtime constant value
   */
  void pushJitVariable(RuntimeConstant &RC) {
    PendingJitVariables.emplace_back(RC);
  }

  /**
   * @brief Register a lambda for JIT compilation
   * 
   * @param LambdaType Type identifier for the lambda
   */
  void registerLambda(const char *LambdaType) {
    const StringRef LambdaTypeRef{LambdaType};
    PROTEUS_DBG(Logger::logs("proteus")
               << "=> RegisterLambda " << LambdaTypeRef << "\n");
    
    // Copy PendingJitVariables if there were changed, otherwise the runtime
    // values for the lambda definition have not changed.
    if (!PendingJitVariables.empty()) {
      LambdaMap[LambdaTypeRef] = {
        LambdaTypeRef,
        SmallVector<RuntimeConstant>{PendingJitVariables}
      };
      PendingJitVariables.clear();
    }
  }

  /**
   * @brief Get JIT variables for a lambda type
   * 
   * @param LambdaTypeRef Lambda type identifier
   * @return Reference to runtime constant values for the lambda
   */
  const SmallVector<RuntimeConstant>& getJitVariables(StringRef LambdaTypeRef) {
    auto It = LambdaMap.find(LambdaTypeRef);
    if (It == LambdaMap.end()) {
      static const SmallVector<RuntimeConstant> EmptyVec;
      return EmptyVec;
    }
    return It->second.CaptureValues;
  }

  /**
   * @brief Match a function name to a registered lambda
   * 
   * @param FnName Function name to match
   * @return Optional reference to matching lambda, or nullopt if not found
   */
  std::optional<std::reference_wrapper<const LambdaInfo>>
  matchLambda(StringRef FnName) const {
    std::string Operator = llvm::demangle(FnName.str());
    std::size_t Sep = Operator.rfind("::operator()");
    if (Sep == std::string::npos) {
      PROTEUS_DBG(Logger::logs("proteus")
                 << "... SKIP ::operator() not found\n");
      return std::nullopt;
    }

    StringRef LambdaType = StringRef{Operator}.slice(0, Sep);
#if PROTEUS_ENABLE_DEBUG
    Logger::logs("proteus")
        << "Operator " << Operator << "\n=> LambdaType to match " << LambdaType
        << "\n";
    Logger::logs("proteus") << "Available Keys\n";
    for (auto &[Key, Val] : LambdaMap) {
      Logger::logs("proteus") << "\tKey: " << Key << "\n";
    }
    Logger::logs("proteus") << "===\n";
#endif

    auto It = LambdaMap.find(LambdaType);
    if (It == LambdaMap.end())
      return std::nullopt;

    return std::cref(It->second);
  }

  /**
   * @brief Check if the registry is empty
   */
  bool empty() const { 
    return FunctionMap.empty() && LambdaMap.empty(); 
  }

private:
  CodeContext() = default;
  ~CodeContext() = default;

  // Pending JIT variables for the next lambda registration
  SmallVector<RuntimeConstant> PendingJitVariables;
  
  // Map from lambda type to lambda information
  std::unordered_map<StringRef, LambdaInfo> LambdaMap;
  
  // Map from function name to function information
  std::unordered_map<std::string, FunctionInfo> FunctionMap;
};

} // namespace proteus

#endif // PROTEUS_CODE_CONTEXT_HPP