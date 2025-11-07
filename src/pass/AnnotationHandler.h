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
// Long form, which uses the annotate attributes in a JSON dict format to
// specify arguments to specialize for. Although it is technically possible for
// users to directly provide the long form notation, the expected usage is that
// they call proteus::jit_arg or proteus::jit_array instrumentation functions,
// esp. since reconstructing the expected JSON form and providing explicit
// runtime type enum values is not ideal.
//
// Both annotations forms populate the global variable llvm.global.annotations
// in the LLVM IR. For split device compilation, the handler also emits a
// JSON manifest file for the host compilation to parse, since there is no other
// direct channel between the split device and host compilation.

namespace proteus {

using namespace llvm;

class AnnotationHandler {
public:
  AnnotationHandler(Module &M);

  void
  parseAnnotations(MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap,
                   const DenseMap<Value *, GlobalVariable *> &StubToKernelMap,
                   bool ForceJitAnnotateAll);

  void parseManifestFileAnnotations(
      const DenseMap<Value *, GlobalVariable *> &StubToKernelMap,
      MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap);

private:
  Module &M;
  ProteusTypes Types;

  SmallString<64> getUniqueManifestFilename();

  void appendToGlobalAnnotations(SmallVector<Constant *> &NewAnnotations);

  Constant *createJitAnnotation(
      Function *F, const SmallSetVector<RuntimeConstantInfo, 16> &ConstantArgs);

  void createDeviceManifestFile(
      DenseMap<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap);

  void parseJitArgAnnotations(
      SmallPtrSetImpl<Function *> &JitArgAnnotations,
      DenseMap<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap);

  void parseJitArrayAnnotations(
      SmallPtrSetImpl<Function *> &JitArrayAnnotations,
      DenseMap<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap);

  void parseJitObjectAnnotations(
      SmallPtrSetImpl<Function *> &JitObjectAnnotations,
      DenseMap<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap);

  void parseJitGlobalAnnotations(
      GlobalVariable *GlobalAnnotations,
      MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap);

  void removeJitGlobalAnnotations();
};

} // namespace proteus

#endif
