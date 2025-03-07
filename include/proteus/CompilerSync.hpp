#ifndef PROTEUS_SYNC_COMPILER_HPP
#define PROTEUS_SYNC_COMPILER_HPP

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include "proteus/CompilationTask.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

class CompilerSync {
public:
  static CompilerSync &instance() {
    static CompilerSync Singleton;
    return Singleton;
  }

  std::unique_ptr<MemoryBuffer> compile(CompilationTask &&CT) {
    return CT.compile();
  }

private:
  CompilerSync() {}

  ~CompilerSync() {}
};

} // namespace proteus

#endif
