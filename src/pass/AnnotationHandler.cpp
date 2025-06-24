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

#include "proteus/Error.h"

#include "AnnotationHandler.h"
#include "Helpers.h"
#include "Types.h"

namespace proteus {

using namespace llvm;

/// If V ultimately came from a store of an argument into an alloca or
/// addrspacecast, return that argument, otherwise return nullptr.
static Argument *getOriginatingArgument(Value *V) {
  // Strip intermittent casts.
  auto StripAllCasts = [](Value *V) {
    while (auto *C = dyn_cast<CastInst>(V))
      V = C->getOperand(0);

    return V;
  };

  V = StripAllCasts(V);

  if (auto *Arg = dyn_cast<Argument>(V)) {
    return Arg;
  }

  // Find the argument by walking a load of a stack slot, which is the
  // typical O0 code generation. An alternative would be to run mem2reg but
  // that will affect the original module.
  auto *LI = dyn_cast<LoadInst>(V);
  if (!LI)
    return nullptr;

  StoreInst *SingleStore = nullptr;
  Value *LoadSource = LI->getPointerOperand();
  for (auto *U : LoadSource->users()) {
    auto *SI = dyn_cast<StoreInst>(U);
    if (!SI || SI->getPointerOperand() != LoadSource)
      continue;

    if (SingleStore != nullptr)
      PROTEUS_FATAL_ERROR("Expected single store");

    SingleStore = SI;
  }

  Value *Val = StripAllCasts(SingleStore->getValueOperand());
  if (auto *Arg = dyn_cast<Argument>(Val))
    return Arg;

  return nullptr;
}

static bool isDeviceKernel(Module &M, const Function *F) {
  auto GetDeviceKernels = []([[maybe_unused]] Module &M) {
    SmallPtrSet<Function *, 16> Kernels;
#if PROTEUS_ENABLE_CUDA
    NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

    if (!MD)
      return Kernels;

    for (auto *Op : MD->operands()) {
      if (Op->getNumOperands() < 2)
        continue;
      MDString *KindID = dyn_cast<MDString>(Op->getOperand(1));
      if (!KindID || KindID->getString() != "kernel")
        continue;

      Function *KernelFn =
          mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
      if (!KernelFn)
        continue;

      Kernels.insert(KernelFn);
    }
#elif PROTEUS_ENABLE_HIP
    for (Function &F : M)
      if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL)
        Kernels.insert(&F);
#endif

    return Kernels;
  };

  static auto KernelSet = GetDeviceKernels(M);

  if (KernelSet.contains(F))
    return true;

  return false;
}

static bool isLambdaFunction(const Function &F) {
  std::string DemangledName = demangle(F.getName().str());
  return StringRef{DemangledName}.contains("'lambda") &&
         StringRef{DemangledName}.contains(")::operator()");
}

AnnotationHandler::AnnotationHandler(Module &M) : M(M), Types(M) {}

void AnnotationHandler::parseAnnotations(
    MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap) {
  // First parse any proteus::jit_arg annotations and append them to global
  // annotations.
  SmallPtrSet<Function *, 32> JitArgAnnotations;
  for (auto &F : M.getFunctionList()) {
    std::string DemangledName = demangle(F.getName().str());
    if (StringRef{DemangledName}.contains("proteus::jit_arg"))
      JitArgAnnotations.insert(&F);
  }

  if (!JitArgAnnotations.empty())
    parseJitArgAnnotations(JitArgAnnotations);

  // Last, parse global annotations, either created throught attributes or the
  // parsed proteus::jit_arg interface.
  auto *GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  if (!JitArgAnnotations.empty() && !GlobalAnnotations)
    PROTEUS_FATAL_ERROR("Expected llvm.global.annotations global variable "
                        "after proteus::jit_arg annotations are parsed.");

  if (GlobalAnnotations)
    parseAttributeAnnotations(GlobalAnnotations, JitFunctionInfoMap);
}

void AnnotationHandler::parseManifestFileAnnotations(
    const DenseMap<Value *, GlobalVariable *> &StubToKernelMap,
    MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap) {
  // Parse the JSON manifest from device compilation, if there exists one, and
  // update JitFunctionInfoMap.
  SmallString<64> UniqueFilename = getUniqueManifestFilename();
  // If there is no manifest file, return early.
  if (!sys::fs::exists(UniqueFilename))
    return;

  auto ErrorOrManifestBuf = MemoryBuffer::getFile(UniqueFilename);
  if (!ErrorOrManifestBuf)
    PROTEUS_FATAL_ERROR("Error reading json manifest file " + UniqueFilename);

  std::unique_ptr<MemoryBuffer> ManifestBuf = std::move(*ErrorOrManifestBuf);
  auto ExpectedJsonValue = json::parse(ManifestBuf->getBuffer());
  if (auto E = ExpectedJsonValue.takeError())
    PROTEUS_FATAL_ERROR("Failed to parse json: " + toString(std::move(E)));

  json::Value ManifestValue = *ExpectedJsonValue;
  json::Object *Manifest = ManifestValue.getAsObject();
  if (!Manifest)
    PROTEUS_FATAL_ERROR("Failed to parse json: manifest object");

  json::Array *KernelArray = Manifest->getArray("manifest");
  if (!KernelArray)
    PROTEUS_FATAL_ERROR("Failed to parse json: kernel array");

  for (auto &Entry : *KernelArray) {
    json::Object *KernelObject = Entry.getAsObject();
    if (!KernelObject)
      PROTEUS_FATAL_ERROR("Failed parsing json: kernel object");

    auto OptionalKernelSym = KernelObject->getString("symbol");
    if (!OptionalKernelSym)
      PROTEUS_FATAL_ERROR("Failed parsing json: function symbol");

    StringRef KernelSym = *OptionalKernelSym;

    json::Array *JitArgs = KernelObject->getArray("args");
    if (!JitArgs)
      PROTEUS_FATAL_ERROR("Failed parsing json: jit args");

    // Find the device stub function searching the StubToKernelMap.
    Function *F = nullptr;
    for (auto [Stub, KernelSymGV] : StubToKernelMap) {
      ConstantDataArray *CDA =
          dyn_cast<ConstantDataArray>(KernelSymGV->getInitializer());
      if (!CDA)
        PROTEUS_FATAL_ERROR("Expected ConstantDataArray");
      if (!CDA->isString())
        PROTEUS_FATAL_ERROR(
            "Expected string constant storing the kernel symbol");

      // Get the value as a CString to avoid including an extra null
      // terminator character that spuriously fails the following comparison.
      StringRef MappedKernelSym = CDA->getAsCString();
      if (MappedKernelSym == KernelSym) {
        F = dyn_cast<Function>(Stub);
        if (!F)
          PROTEUS_FATAL_ERROR("Expected stub function");
        break;
      }
    }

    if (!F)
      PROTEUS_FATAL_ERROR("Expected device stub Function for kernel sym " +
                          KernelSym);

    // Update the JitFunctionInfoMap for the stub function proxying the
    // kernel.
    auto &JFI = JitFunctionInfoMap[F];
    for (auto It : *JitArgs) {
      auto OptionalArgNo = It.getAsInteger();
      if (!OptionalArgNo)
        PROTEUS_FATAL_ERROR("Error parsing json: jit arg no");

      int ArgNo = *OptionalArgNo;
      if (!JFI.ConstantArgs.insert(ArgNo))
        PROTEUS_FATAL_ERROR(
            "Duplicate JIT annotation for argument (0-index): " +
            std::to_string(ArgNo));
    }
  }

  std::remove(UniqueFilename.c_str());
}

SmallString<64> AnnotationHandler::getUniqueManifestFilename() {
  auto TmpPath = std::filesystem::temp_directory_path();

  return {TmpPath.string(), "/", "proteus-device-manifest-", getUniqueFileID(M),
          ".json"};
}

void AnnotationHandler::appendToGlobalAnnotations(
    SmallVector<Constant *> &NewAnnotations) {
  auto *GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  SmallVector<Constant *> CurrentAnnotations;
  // If there is an llvm.global.annotations global variable we get the info
  // and append, otherwise we need to create it.
  if (GlobalAnnotations) {
    if (Constant *Init = GlobalAnnotations->getInitializer()) {
      unsigned N = Init->getNumOperands();
      CurrentAnnotations.reserve(N + 1);
      for (unsigned I = 0; I != N; ++I) {
        CurrentAnnotations.push_back(cast<Constant>(Init->getOperand(I)));
      }
    }
    GlobalAnnotations->eraseFromParent();
  }

  CurrentAnnotations.insert(CurrentAnnotations.end(), NewAnnotations.begin(),
                            NewAnnotations.end());

  ArrayType *AT =
      ArrayType::get(Types.GlobalAnnotationEltTy, CurrentAnnotations.size());
  Constant *Init = ConstantArray::get(AT, CurrentAnnotations);
  auto *AnnotationsGV = new GlobalVariable(M, Init->getType(), false,
                                           GlobalValue::AppendingLinkage, Init,
                                           "llvm.global.annotations");
  AnnotationsGV->setSection("llvm.metadata");
}

Constant *
AnnotationHandler::createJitAnnotation(Function *F,
                                       SmallVector<int> &ConstantArgs) {
  // llvm.global.annotations entry format:
  //  ptr: (addrspace 1) Function pointer
  //  ptr: (addrspace 4) Annotations string
  //  ptr: (addrspace 4) Source file string
  //  i32: Line number,
  //  ptr: (addrspace 1) Arguments pointer
  IRBuilder<> IRB{M.getContext()};
  constexpr size_t NumElts = 5;
  Constant *AnnotationVals[NumElts];

  if (isDeviceCompilation(M)) {
    constexpr unsigned GlobalAddressSpace = 1;
    constexpr unsigned ConstantAddressSpace = 4;
    AnnotationVals[0] = cast<Constant>(
        IRB.CreateAddrSpaceCast(F, IRB.getPtrTy(GlobalAddressSpace)));
    AnnotationVals[1] =
        IRB.CreateGlobalString("jit", ".str", ConstantAddressSpace, &M);
    AnnotationVals[2] = IRB.CreateGlobalString(M.getSourceFileName(), "",
                                               ConstantAddressSpace, &M);
  } else {
    AnnotationVals[0] = F;
    AnnotationVals[1] = IRB.CreateGlobalString("jit", ".str", 0, &M);
    AnnotationVals[2] =
        IRB.CreateGlobalString(M.getSourceFileName(), "", 0, &M);
  }
  // We don't know the line number, hence we store 0.
  AnnotationVals[3] = IRB.getInt32(0);

  // Create the struct to store the JIT argument numbers.
  SmallVector<Type *> ArgInfo{ConstantArgs.size(), Types.Int32Ty};
  StructType *ArgEltTy = StructType::get(M.getContext(), ArgInfo);
  SmallVector<Constant *> ArgConsts;
  for (int ArgNo : ConstantArgs)
    // We add 1 to the ArgNo to create the 1-index argument number that global
    // annotations expect.
    // TODO: Maybe require 0-indexing to avoid this?
    ArgConsts.push_back(ConstantInt::get(Types.Int32Ty, ArgNo + 1));
  Constant *ArgInit = ConstantStruct::get(ArgEltTy, ArgConsts);

  GlobalVariable *ArgsInfoGV = nullptr;
  if (isDeviceCompilation(M)) {
    constexpr unsigned GlobalAddressSpace = 1;
    ArgsInfoGV = new GlobalVariable(
        M, ArgInit->getType(), true, GlobalValue::PrivateLinkage, ArgInit,
        ".args", nullptr, llvm::GlobalValue::NotThreadLocal,
        GlobalAddressSpace);
  } else {
    ArgsInfoGV =
        new GlobalVariable(M, ArgInit->getType(), true,
                           GlobalValue::PrivateLinkage, ArgInit, ".args");
  }
  AnnotationVals[4] = ArgsInfoGV;

  Constant *NewAnnotation = ConstantStruct::get(
      Types.GlobalAnnotationEltTy, ArrayRef{AnnotationVals, NumElts});

  return NewAnnotation;
}

void AnnotationHandler::createDeviceManifestFile(
    DenseMap<Function *, SmallSetVector<int, 16>> &JitArgs) {
  // Emit JSON file manifest which contains the kernel symbol and
  // JIT-annotated arguments.
  SmallString<64> UniqueFilename = getUniqueManifestFilename();
  std::error_code EC;
  raw_fd_ostream OS(UniqueFilename, EC, sys::fs::OF_Text);
  if (EC)
    PROTEUS_FATAL_ERROR("Error opening device manifest file " + EC.message());

  json::Object ManifestInfo;
  json::Array KernelArray;

  for (auto [F, ConstantArgs] : JitArgs) {
    json::Object KernelObject;
    KernelObject["symbol"] = F->getName();

    json::Array JitArgNos;
    for (auto ArgNo : ConstantArgs) {
      JitArgNos.push_back(ArgNo);
    }
    KernelObject["args"] = std::move(JitArgNos);

    KernelArray.push_back(std::move(KernelObject));
  }

  ManifestInfo["manifest"] = std::move(KernelArray);

  OS << formatv("{0:2}", json::Value(std::move(ManifestInfo)));
  OS.close();
}

void AnnotationHandler::parseJitArgAnnotations(
    SmallPtrSetImpl<Function *> &JitArgAnnotations) {
  // Iterate over all proteus::jit_arg annotations and store the information
  // in the JitArgs map.
  DenseMap<Function *, SmallSetVector<int, 16>> JitArgs;
  SmallVector<CallBase *> CallsToDelete;
  for (Function *AnnotationF : JitArgAnnotations) {
    for (User *Usr : AnnotationF->users()) {
      CallBase *CB = dyn_cast<CallBase>(Usr);
      if (!CB)
        continue;

      if (CB->getNumUses() != 0)
        PROTEUS_FATAL_ERROR("Expected zero uses of the annotation function");

      CallsToDelete.push_back(CB);

      Function *JitFunction = CB->getFunction();
      assert(CB->arg_size() == 1 && "Expected single argument");
      auto *V = CB->getArgOperand(0);

      Argument *Arg = getOriginatingArgument(V);
      if (!Arg)
        PROTEUS_FATAL_ERROR(
            "Expected non-null argument. Possible cause: proteus::jit_arg "
            "argument is not an argument of the enclosing function.");

      auto &ConstantArgs = JitArgs[JitFunction];
      if (!ConstantArgs.insert(Arg->getArgNo()))
        PROTEUS_FATAL_ERROR("Duplicate argument number found: " +
                            std::to_string(Arg->getArgNo()));
    }
  }

  // Remove JitArg annotations and their calls as they are not needed anymore.
  // Also removing them avoids errors with LTO and O0 compilation: LTO will
  // find those annotations and attempt to create a manifest which will cause
  // an error since it is not backed by a source filename.
  for (CallBase *CB : CallsToDelete)
    CB->eraseFromParent();

  for (Function *AnnotationF : JitArgAnnotations)
    AnnotationF->eraseFromParent();

  if (verifyModule(M, &errs()))
    PROTEUS_FATAL_ERROR("Broken JIT module found, compilation aborted!");

  // Sort argument numbers for determinism.
  SmallVector<Constant *> NewJitAnnotations;
  for (auto &[F, ConstantArgs] : JitArgs) {
    SmallVector<int> SortedArgs{ConstantArgs.begin(), ConstantArgs.end()};
    std::sort(SortedArgs.begin(), SortedArgs.end());
    ConstantArgs = {SortedArgs.begin(), SortedArgs.end()};

    NewJitAnnotations.push_back(createJitAnnotation(F, SortedArgs));
  }

  // We append to global annotations the parsed information from the manifest
  // file. This is needed for HIP LTO because it uses global annotations to
  // identify kernels.
  appendToGlobalAnnotations(NewJitAnnotations);
  // If this is device compilation the pass emits a JSON file that stores this
  // information for the host compilation pass to parse for instrumentation.
  // The JSON file is uniquely named using the TU unique file ID.
  if (isDeviceCompilation(M))
    createDeviceManifestFile(JitArgs);
}

void AnnotationHandler::parseAttributeAnnotations(
    GlobalVariable *GlobalAnnotations,
    MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap) {
  auto *Array = cast<ConstantArray>(GlobalAnnotations->getOperand(0));
  DEBUG(Logger::logs("proteus-pass") << "Annotation Array " << *Array << "\n");
  for (unsigned int I = 0; I < Array->getNumOperands(); I++) {
    auto *Entry = cast<ConstantStruct>(Array->getOperand(I));
    DEBUG(Logger::logs("proteus-pass") << "Entry " << *Entry << "\n");

    auto *Fn = dyn_cast<Function>(Entry->getOperand(0)->stripPointerCasts());

    assert(Fn && "Expected function in entry operands");

    // Check the annotated function is a kernel function or a device
    // lambda.
    if (isDeviceCompilation(M)) {
      if (!isDeviceKernel(M, Fn) && !isLambdaFunction(*Fn))
        PROTEUS_FATAL_ERROR(
            std::string{} + __FILE__ + ":" + std::to_string(__LINE__) +
            " => Expected the annotated Fn " + Fn->getName() + " (" +
            demangle(Fn->getName().str()) +
            ") to be a kernel function or device lambda function!");
    }

    if (JitFunctionInfoMap.contains(Fn)) {
      DEBUG(Logger::logs("proteus-pass")
            << "Warning: Duplicate jit annotation for Fn " + Fn->getName() +
                   "\n");
      continue;
    }

    DEBUG(Logger::logs("proteus-pass")
          << "JIT Function " << Fn->getName() << "\n");

    auto *Annotation =
        cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));

    DEBUG(Logger::logs("proteus-pass")
          << "Annotation " << Annotation->getAsCString() << "\n");

    // Needs CString for comparison to strip extract null byte character.
    if (Annotation->getAsCString().compare("jit"))
      continue;

    JitFunctionInfo JFI;

    if (Entry->getOperand(4)->isNullValue()) {
      JFI.ConstantArgs = {};
      JitFunctionInfoMap[Fn] = JFI;
      continue;
    }

    DEBUG(Logger::logs("proteus-pass")
          << "AnnotArgs " << *Entry->getOperand(4)->getOperand(0) << "\n");
    DEBUG(Logger::logs("proteus-pass")
          << "Type AnnotArgs "
          << *Entry->getOperand(4)->getOperand(0)->getType() << "\n");
    auto *AnnotArgs = cast<ConstantStruct>(Entry->getOperand(4)->getOperand(0));

    SmallSetVector<int, 16> JitArgs;
    for (unsigned int J = 0; J < AnnotArgs->getNumOperands(); ++J) {
      uint64_t ArgNo;
      Constant *C = AnnotArgs->getOperand(J)->stripPointerCasts();
      // We parse either numeric arguments in short form or the key-value pairs
      // in long form.
      if (isa<ConstantInt>(C)) {
        auto *Index = cast<ConstantInt>(AnnotArgs->getOperand(J));
        ArgNo = Index->getValue().getZExtValue();
      } else {
        auto *CDA = dyn_cast<ConstantDataArray>(C->getOperand(0));
        if (!CDA)
          PROTEUS_FATAL_ERROR("Expected constant data array string");

        StringRef KVStr = CDA->getAsCString();
        auto [K, V] = KVStr.split("=");
        if (V.empty())
          PROTEUS_FATAL_ERROR("Wrong format: " + KVStr);

        V.getAsInteger(10, ArgNo);
      }

      if (ArgNo > Fn->arg_size())
        PROTEUS_FATAL_ERROR(
            Twine("Error: JIT annotation runtime constant argument " +
                  std::to_string(ArgNo) +
                  " is greater than number of arguments " +
                  std::to_string(Fn->arg_size()))
                .str()
                .c_str());
      // TODO: think about types, -1 to convert to 0-start index.
      if (!JitArgs.insert(ArgNo - 1))
        PROTEUS_FATAL_ERROR(
            "Duplicate JIT annotation for argument (0-index): " +
            std::to_string(ArgNo - 1));
    }

    // Sort JFI.ConstantArgs for determinism.
    SmallVector<int> SortedArgs{JitArgs.begin(), JitArgs.end()};
    std::sort(SortedArgs.begin(), SortedArgs.end());
    JFI.ConstantArgs = {SortedArgs.begin(), SortedArgs.end()};
    JitFunctionInfoMap[Fn] = JFI;
  }
}

} // namespace proteus
