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

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include "CompilerInterfaceTypes.h"
#include "JitCache.hpp"
#include "JitEngine.hpp"

namespace proteus {

class JitEngineHost : JitEngine {
public:
  std::unique_ptr<llvm::orc::LLJIT> LLJITPtr;
  llvm::ExitOnError ExitOnErr;

  static JitEngineHost &instance();

  static void
  dumpSymbolInfo(const llvm::object::ObjectFile &loadedObj,
                 const llvm::RuntimeDyld::LoadedObjectInfo &objInfo);
  static void notifyLoaded(llvm::orc::MaterializationResponsibility &R,
                           const llvm::object::ObjectFile &Obj,
                           const llvm::RuntimeDyld::LoadedObjectInfo &LOI);
  ~JitEngineHost();

  llvm::Expected<llvm::orc::ThreadSafeModule>
  specializeBitcode(llvm::StringRef FnName, llvm::StringRef Suffix,
                    llvm::StringRef IR, RuntimeConstant *RC,
                    int NumRuntimeConstants);

  void *compileAndLink(llvm::StringRef FnName, char *IR, int IRSize,
                       RuntimeConstant *RC, int NumRuntimeConstants);

private:
  JitEngineHost(int argc, char *argv[]);
  void addStaticLibrarySymbols();
  JitCache<void *> CodeCache;
};

} // namespace proteus

#endif
