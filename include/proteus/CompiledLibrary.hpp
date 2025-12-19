#ifndef PROTEUS_COMPILED_LIBRARY_HPP
#define PROTEUS_COMPILED_LIBRARY_HPP

#include <memory>
#include <string>

namespace llvm {
class MemoryBuffer;
namespace orc {
class JITDylib;
} // namespace orc
} // namespace llvm

namespace proteus {

using namespace llvm;

struct CompiledLibrary {
  // Compiled library holds a relocatable object or a dynamic library, indicated
  // by IsDynLib.
  std::unique_ptr<MemoryBuffer> ObjectModule;
  std::string DynLibPath;
  bool IsDynLib;
  bool IsLoaded = false;
  // JitDyLib holds a pointer to the ORC JIT dynamic library context for the
  // host JIT engine.
  llvm::orc::JITDylib *JitDyLib = nullptr;

  CompiledLibrary(std::unique_ptr<MemoryBuffer> ObjectModule);

  CompiledLibrary(const std::string &Path);

  ~CompiledLibrary();

  bool isSharedObject() const { return IsDynLib; }

  bool isStaticObject() const { return !IsDynLib; }
};

} // namespace proteus

#endif
