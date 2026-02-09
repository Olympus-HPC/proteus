#ifndef PROTEUS_SYNC_COMPILER_H
#define PROTEUS_SYNC_COMPILER_H

#include "proteus/impl/CompilationTask.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Hashing.h"

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

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
