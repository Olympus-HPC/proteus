//===-- JitEngineHost.h -- JIT Engine for CPU header --===//
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

#ifndef PROTEUS_JITENGINEHOST_H
#define PROTEUS_JITENGINEHOST_H

#include <functional>
#include <optional>
#include <string>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/Init.h"
#include "proteus/impl/Caching/MemoryCache.h"
#include "proteus/impl/Caching/ObjectCacheChain.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/JitEngine.h"

namespace proteus {

using namespace llvm;

class JitEngineHost : public JitEngine {
public:
  std::unique_ptr<orc::LLJIT> LLJITPtr;
  ExitOnError ExitOnErr;

  static JitEngineHost &instance();

  static void dumpSymbolInfo(const object::ObjectFile &LoadedObj,
                             const RuntimeDyld::LoadedObjectInfo &ObjInfo);
  static void notifyLoaded(orc::MaterializationResponsibility &R,
                           const object::ObjectFile &Obj,
                           const RuntimeDyld::LoadedObjectInfo &LOI);
  ~JitEngineHost();

  void specializeIR(Module &M, StringRef FnName, StringRef Suffix,
                    ArrayRef<RuntimeConstant> RCArray);

  void *compileAndLink(StringRef FnName, char *IR, int IRSize, void **Args,
                       ArrayRef<RuntimeConstantInfo *> RCInfoArray);

  std::unique_ptr<MemoryBuffer> compileOnly(Module &M,
                                            bool DisableIROpt = false);

  void loadCompiledLibrary(CompiledLibrary &Library);

  void *getFunctionAddress(StringRef FnName, CompiledLibrary &Library);

  void init() {
    CacheChain = std::make_unique<ObjectCacheChain>("JitEngineHost");
  }

  void finalize() {
    if (CacheChain)
      CacheChain->finalize();
  }

  std::optional<std::reference_wrapper<ObjectCacheChain>> getLibraryCache() {
    if (!Config::get().ProteusUseStoredCache || !CacheChain)
      return std::nullopt;
    return std::ref(*CacheChain);
  }

private:
  JitEngineHost();
  void addStaticLibrarySymbols();
  MemoryCache<void *> CodeCache{"JitEngineHost"};
  std::unique_ptr<ObjectCacheChain> CacheChain;
};

} // namespace proteus

#endif
