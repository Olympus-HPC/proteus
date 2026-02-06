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

#include "proteus/Caching/MemoryCache.h"
#include "proteus/Caching/ObjectCacheRegistry.h"
#include "proteus/CompiledLibrary.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Config.h"
#include "proteus/Error.h"
#include "proteus/Init.h"
#include "proteus/JitEngine.h"

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

  void initCacheChain() {
    ObjectCacheRegistry::instance().create("JitEngineHost");
  }

  void finalize() {
    if (auto CacheOpt = ObjectCacheRegistry::instance().get("JitEngineHost")) {
      CacheOpt->get().finalize();
    }
  }

private:
  JitEngineHost();
  void addStaticLibrarySymbols();
  MemoryCache<void *> CodeCache{"JitEngineHost"};
};

} // namespace proteus

#endif
