//===-- JitEngineHost.hpp -- JIT Engine for CPU header --===//
//
// Part of Proteus Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the JitEngine interface for dynamic compilation and optimization
// of CPU code.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINEHOST_HPP
#define PROTEUS_JITENGINEHOST_HPP

#include <string>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/JitCache.hpp"
#include "proteus/JitEngine.hpp"

namespace proteus {

using namespace llvm;

class JitEngineHost : public JitEngine {
public:
  std::unique_ptr<orc::LLJIT> LLJITPtr;
  ExitOnError ExitOnErr;

  static JitEngineHost &instance();

  static void dumpSymbolInfo(const object::ObjectFile &loadedObj,
                             const RuntimeDyld::LoadedObjectInfo &objInfo);
  static void notifyLoaded(orc::MaterializationResponsibility &R,
                           const object::ObjectFile &Obj,
                           const RuntimeDyld::LoadedObjectInfo &LOI);
  ~JitEngineHost();

  Expected<orc::ThreadSafeModule>
  specializeIR(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> Ctx,
               StringRef FnName, StringRef Suffix,
               ArrayRef<RuntimeConstant> RCArray);

  void *compileAndLink(StringRef FnName, char *IR, int IRSize, void **Args,
                       ArrayRef<RuntimeConstantInfo *> RCInfoArray);

  void compileOnly(std::unique_ptr<Module> M);

  void *getFunctionAddress(StringRef FnName);

private:
  JitEngineHost();
  void addStaticLibrarySymbols();
  JitCache<void *> CodeCache;
};

} // namespace proteus

#endif
