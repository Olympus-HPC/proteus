#ifndef PROTEUS_SYNC_COMPILER_H
#define PROTEUS_SYNC_COMPILER_H

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include "proteus/CompilationTask.h"
#include "proteus/Debug.h"
#include "proteus/Hashing.h"

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
