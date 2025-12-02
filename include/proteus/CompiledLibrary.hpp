#ifndef PROTEUS_COMPILED_LIBRARY_HPP
#define PROTEUS_COMPILED_LIBRARY_HPP

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

using namespace llvm;

struct CompiledLibrary {
  // Compiled library holds a relocatable object or a dynamic library, indicated
  // by IsDynLib.
  std::unique_ptr<MemoryBuffer> ObjectModule;
  SmallString<128> DynLibPath;
  bool IsDynLib;
  bool IsLoaded = false;
  // JitDyLib holds a pointer to the ORC JIT dynamic library context for the
  // host JIT engine.
  llvm::orc::JITDylib *JitDyLib = nullptr;

  CompiledLibrary(std::unique_ptr<MemoryBuffer> ObjectModule)
      : ObjectModule(std::move(ObjectModule)), IsDynLib(false) {}

  CompiledLibrary(const SmallString<128> &Path)
      : DynLibPath{Path}, IsDynLib(true) {}

  bool isSharedObject() const { return IsDynLib; }

  bool isStaticObject() const { return !IsDynLib; }
};

} // namespace proteus

#endif
