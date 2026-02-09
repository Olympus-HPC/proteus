#include <memory>
#include <string>

#include <llvm/Support/MemoryBuffer.h>

#include "proteus/CompiledLibrary.h"

namespace proteus {

using namespace llvm;

CompiledLibrary::CompiledLibrary(std::unique_ptr<MemoryBuffer> ObjectModule)
    : ObjectModule(std::move(ObjectModule)), IsDynLib(false) {}

CompiledLibrary::CompiledLibrary(const std::string &Path)
    : DynLibPath{Path}, IsDynLib(true) {}

CompiledLibrary::~CompiledLibrary() = default;
} // namespace proteus
