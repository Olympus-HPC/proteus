#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>

#include "proteus/Error.h"

using namespace llvm;

void linkModules(Module &Dest,
                 SmallVector<std::unique_ptr<Module>> &LinkedModules) {
  Linker IRLinker(Dest);
  for (auto &Mod : LinkedModules) {
    if (IRLinker.linkInModule(std::move(Mod)))
      PROTEUS_FATAL_ERROR("Linking failed");
  }
}