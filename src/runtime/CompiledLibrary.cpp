
#include "proteus/impl/CompiledLibrary.h"

#include <llvm/Support/MemoryBuffer.h>

#include <memory>
#include <string>

namespace proteus {

using namespace llvm;

CompiledLibrary::CompiledLibrary(std::unique_ptr<MemoryBuffer> ObjectModule)
    : ObjectModule(std::move(ObjectModule)), IsDynLib(false) {}

CompiledLibrary::CompiledLibrary(const std::string &Path)
    : DynLibPath{Path}, IsDynLib(true) {}

CompiledLibrary::~CompiledLibrary() = default;
} // namespace proteus
