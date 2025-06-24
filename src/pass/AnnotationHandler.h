#ifndef PROTEUS_PASS_ANNOTATIONS_PARSER_H
#define PROTEUS_PASS_ANNOTATIONS_PARSER_H

#include <llvm/ADT/SetVector.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>

#include "Helpers.h"
#include "Types.h"

// The annotation handler supports two types of attribute-based annotations:
// Short form, which uses an annotate attribute to specify an 1-indexed,
// comma-separated list of scalar arguments as integers to specialize for:
//
//        __attibute__((annotate("jit", [<argument list>])))
//
// Long form, which uses the annotate attribute to specify arguments to
// specialize but also supports arrays. The long form uses key-value pairs:
// arg=<1-index argument number>:
//
//        __attribute__((annotate("jit", ["arg=<N>"]*)))
//
// Those annotations populate the llvm.global.annotations global variable in the
// LLVM IR.
//
// Besides attribute-based annotations, the handler supports annotations through
// the C++ API, namely proteus::jit_arg. The handler parses the IR and populates
// llvm.global.annotations accordingly. For split device compilation, it emits a
// JSON manifest file that the host compilation parses.

namespace proteus {

using namespace llvm;

class AnnotationHandler {
public:
  AnnotationHandler(Module &M);

  void
  parseAnnotations(MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap);

  void parseManifestFileAnnotations(
      const DenseMap<Value *, GlobalVariable *> &StubToKernelMap,
      MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap);

private:
  Module &M;
  ProteusTypes Types;

  SmallString<64> getUniqueManifestFilename();

  void appendToGlobalAnnotations(SmallVector<Constant *> &NewAnnotations);

  Constant *createJitAnnotation(Function *F, SmallVector<int> &ConstantArgs);

  void createDeviceManifestFile(
      DenseMap<Function *, SmallSetVector<int, 16>> &JitArgs);

  void parseJitArgAnnotations(SmallPtrSetImpl<Function *> &JitArgAnnotations);

  void parseAttributeAnnotations(
      GlobalVariable *GlobalAnnotations,
      MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap);
};

} // namespace proteus

#endif
