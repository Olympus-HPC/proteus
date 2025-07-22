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
#include "proteus/RuntimeConstantTypeHelpers.h"

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
  // First parse any proteus::jit_arg or proteus::jit_array annotations and
  // append them to global annotations.
  SmallPtrSet<Function *, 32> JitArgAnnotations;
  SmallPtrSet<Function *, 32> JitArrayAnnotations;
  for (auto &F : M.getFunctionList()) {
    std::string DemangledName = demangle(F.getName().str());
    if (StringRef{DemangledName}.contains("proteus::jit_arg"))
      JitArgAnnotations.insert(&F);
    if (StringRef{DemangledName}.contains("proteus::jit_array"))
      JitArrayAnnotations.insert(&F);
  }

  DenseMap<Function *, SmallSetVector<RuntimeConstantInfo, 16>> RCInfoMap;
  if (!JitArgAnnotations.empty())
    parseJitArgAnnotations(JitArgAnnotations, RCInfoMap);

  if (!JitArrayAnnotations.empty())
    parseJitArrayAnnotations(JitArrayAnnotations, RCInfoMap);

  SmallVector<Constant *> NewJitAnnotations;
  for (auto &[F, RCInfos] : RCInfoMap)
    NewJitAnnotations.push_back(createJitAnnotation(F, RCInfos));

  // We append to global annotations the parsed information. This is to have and
  // a common place to store information and also needed for HIP LTO because it
  // uses global annotations to identify kernels.
  if (!NewJitAnnotations.empty())
    appendToGlobalAnnotations(NewJitAnnotations);
  // If this is device compilation the pass emits a JSON file that stores this
  // information for the host compilation pass to parse for instrumentation.
  // The JSON file is uniquely named using the TU unique file ID.
  if (isDeviceCompilation(M))
    if (!RCInfoMap.empty())
      createDeviceManifestFile(RCInfoMap);

  // Last, parse global annotations, either created throught attributes or the
  // parsed proteus::jit_arg, proteus::jit_array interfaces.
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

    json::Array *JsonRCInfoArray = KernelObject->getArray("rc");
    if (!JsonRCInfoArray)
      PROTEUS_FATAL_ERROR("Failed parsing json: jit args");
    // Update the JitFunctionInfoMap for the stub function proxying the
    // kernel.
    auto &JFI = JitFunctionInfoMap[F];
    for (auto J : *JsonRCInfoArray) {
      auto CreateRuntimeConstantInfo = [&F](json::Object *JsonRCDict) {
        json::Value *ArgV = JsonRCDict->get("arg");
        if (!ArgV)
          PROTEUS_FATAL_ERROR("Expected arg in runtime constant info");
        auto OptArgNo = ArgV->getAsInteger();
        int32_t ArgNo = *OptArgNo;

        json::Value *TypeV = JsonRCDict->get("type");
        if (!TypeV)
          PROTEUS_FATAL_ERROR("Expected type in runtime constant info");
        auto OptRCType = TypeV->getAsInteger();
        RuntimeConstantType RCType =
            static_cast<RuntimeConstantType>(*OptRCType);

        Type *ArgType = F->getArg(ArgNo)->getType();
        RuntimeConstantType ExpectedRCType =
            convertTypeToRuntimeConstantType(ArgType);
        if (ExpectedRCType != RCType) {
          // For GPU code, long double decays to double but the host code still
          // pushes a long double to kernel args. Update the RCType to avoid
          // data marshaling errors.
          if (ExpectedRCType == RuntimeConstantType::LONG_DOUBLE &&
              RCType == RuntimeConstantType::DOUBLE)
            RCType = RuntimeConstantType::LONG_DOUBLE;
          else if (ExpectedRCType == RuntimeConstantType::PTR &&
                   RCType == RuntimeConstantType::ARRAY) {
            // This is a valid check since there is no way we can distinguish a
            // PTR from an ARRAY just by looking at the function argument type.
          } else
            PROTEUS_FATAL_ERROR("Type mismatch, function " + F->getName() +
                                " for argument (0-index) " +
                                std::to_string(ArgNo) + " expected " +
                                toString(ExpectedRCType) + " but found " +
                                toString(RCType));
        }

        if (RCType == RuntimeConstantType::ARRAY) {
          json::Value *EltTypeV = JsonRCDict->get("elt-type");
          auto OptEltType = EltTypeV->getAsInteger();
          if (!OptEltType)
            PROTEUS_FATAL_ERROR("Expected elt-type in runtime constant info");
          RuntimeConstantType EltType =
              static_cast<RuntimeConstantType>(*OptEltType);

          json::Value *NumEltsV = JsonRCDict->get("num-elts");
          json::Value *NumEltsPosV = JsonRCDict->get("num-elts-pos");
          json::Value *NumEltsTypeV = JsonRCDict->get("num-elts-type");

          // Check for semantic errors.
          if (NumEltsV && NumEltsPosV)
            PROTEUS_FATAL_ERROR(
                "Setting both NumElts and NumEltsPos is incorrect");

          if (!NumEltsV && !NumEltsPosV)
            PROTEUS_FATAL_ERROR("Either NumElts or NumEltsPos must be set");

          if (NumEltsPosV && !NumEltsTypeV)
            PROTEUS_FATAL_ERROR("NumEltsType must be set if NumEltsPos is set");

          if (!NumEltsPosV && NumEltsTypeV)
            PROTEUS_FATAL_ERROR("NumEltsPos must be set if NumEltsType is set");

          if (NumEltsV) {
            int32_t NumElts = NumEltsV->getAsInteger().value();
            return RuntimeConstantInfo{RCType, ArgNo, NumElts, EltType};
          }

          int32_t NumEltsPos = NumEltsPosV->getAsInteger().value();
          RuntimeConstantType NumEltsType = static_cast<RuntimeConstantType>(
              NumEltsTypeV->getAsInteger().value());
          return RuntimeConstantInfo{RCType, ArgNo, EltType, NumEltsType,
                                     NumEltsPos};
        }

        return RuntimeConstantInfo{RCType, ArgNo};
      };

      RuntimeConstantInfo RCI = CreateRuntimeConstantInfo(J.getAsObject());
      // RuntimeConstantInfo RCI{RuntimeConstantType::NONE, 0};
      if (!JFI.ConstantArgs.insert(RCI))
        PROTEUS_FATAL_ERROR(
            "Duplicate JIT annotation for argument (0-index): " +
            std::to_string(RCI.ArgInfo.Pos));
    }

    SmallVector<RuntimeConstantInfo> SortedRCInfos{JFI.ConstantArgs.begin(),
                                                   JFI.ConstantArgs.end()};
    std::sort(SortedRCInfos.begin(), SortedRCInfos.end());
    JFI.ConstantArgs = {SortedRCInfos.begin(), SortedRCInfos.end()};
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

Constant *AnnotationHandler::createJitAnnotation(
    Function *F, SmallSetVector<RuntimeConstantInfo, 16> &RCInfos) {
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

  // Create the JIT argument information in long form to include the argument
  // number and possible size. Each RCInfo will create a global variable string
  // with key-value pairs to store the information.
  SmallVector<Type *> RCInfoTypes{RCInfos.size(), Types.PtrTy};
  StructType *RCInfoStructTy = StructType::get(M.getContext(), RCInfoTypes);
  SmallVector<Constant *> RCInfoConsts;
  for (const auto &RCInfo : RCInfos) {
    json::Object JsonRCDict;
    JsonRCDict["arg"] = RCInfo.ArgInfo.Pos + 1;
    JsonRCDict["type"] = static_cast<int32_t>(RCInfo.ArgInfo.Type);

    if (RCInfo.OptArrInfo) {
      JsonRCDict["elt-type"] = static_cast<int32_t>(RCInfo.OptArrInfo->EltType);

      if (RCInfo.OptArrInfo->OptNumEltsRCInfo) {
        JsonRCDict["num-elts-type"] =
            static_cast<int32_t>(RCInfo.OptArrInfo->OptNumEltsRCInfo->Type);
        JsonRCDict["num-elts-pos"] = RCInfo.OptArrInfo->OptNumEltsRCInfo->Pos;
      } else {
        JsonRCDict["num-elts"] = RCInfo.OptArrInfo->NumElts;
      }
    }

    std::string RCInfoStr;
    raw_string_ostream OS{RCInfoStr};
    OS << json::Value(std::move(JsonRCDict));

    if (isDeviceCompilation(M)) {
      constexpr unsigned ConstAddressSpace = 4;
      RCInfoConsts.push_back(
          IRB.CreateGlobalString(RCInfoStr, ".str", ConstAddressSpace, &M));
    } else
      RCInfoConsts.push_back(IRB.CreateGlobalString(RCInfoStr, ".str", 0, &M));
  }
  Constant *RCInfoInit = ConstantStruct::get(RCInfoStructTy, RCInfoConsts);

  GlobalVariable *RCInfoGV = nullptr;
  if (isDeviceCompilation(M)) {
    constexpr unsigned GlobalAddressSpace = 1;
    RCInfoGV = new GlobalVariable(
        M, RCInfoInit->getType(), true, GlobalValue::PrivateLinkage, RCInfoInit,
        ".args", nullptr, llvm::GlobalValue::NotThreadLocal,
        GlobalAddressSpace);
  } else {
    RCInfoGV =
        new GlobalVariable(M, RCInfoInit->getType(), true,
                           GlobalValue::PrivateLinkage, RCInfoInit, ".args");
  }
  AnnotationVals[4] = RCInfoGV;

  Constant *NewAnnotation = ConstantStruct::get(
      Types.GlobalAnnotationEltTy, ArrayRef{AnnotationVals, NumElts});

  return NewAnnotation;
}

void AnnotationHandler::createDeviceManifestFile(
    DenseMap<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  // Emit JSON file manifest which contains the kernel symbol and
  // JIT-annotated arguments.
  SmallString<64> UniqueFilename = getUniqueManifestFilename();
  std::error_code EC;
  raw_fd_ostream OS(UniqueFilename, EC, sys::fs::OF_Text);
  if (EC)
    PROTEUS_FATAL_ERROR("Error opening device manifest file " + EC.message());

  json::Object ManifestInfo;
  json::Array KernelArray;

  for (auto &[F, RCInfos] : RCInfoMap) {
    json::Object KernelObject;
    KernelObject["symbol"] = F->getName();

    json::Array JsonRCInfoArray;
    for (auto &RCInfo : RCInfos) {
      json::Object JsonRCDict;
      JsonRCDict["arg"] = RCInfo.ArgInfo.Pos;
      JsonRCDict["type"] = static_cast<int32_t>(RCInfo.ArgInfo.Type);

      if (RCInfo.OptArrInfo) {
        JsonRCDict["elt-type"] =
            static_cast<int32_t>(RCInfo.OptArrInfo->EltType);

        if (RCInfo.OptArrInfo->OptNumEltsRCInfo) {
          JsonRCDict["num-elts-type"] =
              static_cast<int32_t>(RCInfo.OptArrInfo->OptNumEltsRCInfo->Type);
          JsonRCDict["num-elts-pos"] = RCInfo.OptArrInfo->OptNumEltsRCInfo->Pos;
        } else {
          JsonRCDict["num-elts"] = RCInfo.OptArrInfo->NumElts;
        }
      }

      JsonRCInfoArray.push_back(std::move(JsonRCDict));
    }

    KernelObject["rc"] = std::move(JsonRCInfoArray);

    KernelArray.push_back(std::move(KernelObject));
  }

  ManifestInfo["manifest"] = std::move(KernelArray);

  OS << formatv("{0:2}", json::Value(std::move(ManifestInfo)));
  OS.close();
}

void AnnotationHandler::parseJitArgAnnotations(
    SmallPtrSetImpl<Function *> &JitArgAnnotations,
    DenseMap<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  // Iterate over all proteus::jit_arg annotations and store the information
  // in RCInfoMap.
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

      auto &ConstantArgs = RCInfoMap[JitFunction];
      RuntimeConstantType RCType =
          convertTypeToRuntimeConstantType(Arg->getType());
      int32_t ArgNo = Arg->getArgNo();
      RuntimeConstantInfo RCI{
          RCType,
          ArgNo,
      };
      if (!ConstantArgs.insert(RCI))
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
}

void AnnotationHandler::parseJitArrayAnnotations(
    SmallPtrSetImpl<Function *> &JitArrayAnnotations,
    DenseMap<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  // Iterate over all proteus::jit_array annotations and store the information
  // in RCInfoMap.
  SmallVector<CallBase *> CallsToDelete;
  for (Function *AnnotationF : JitArrayAnnotations) {
    for (User *Usr : AnnotationF->users()) {
      CallBase *CB = dyn_cast<CallBase>(Usr);
      if (!CB)
        continue;

      if (CB->getNumUses() != 0)
        PROTEUS_FATAL_ERROR("Expected zero uses of the annotation function");

      CallsToDelete.push_back(CB);

      Function *JitFunction = CB->getFunction();
      assert(CB->arg_size() == 3 &&
             "Expected three arguments (value, size, element)");
      auto *V = CB->getArgOperand(0);
      auto *Size = CB->getArgOperand(1);
      auto *Elt = CB->getArgOperand(2);

      Argument *Arg = getOriginatingArgument(V);
      if (!Arg)
        PROTEUS_FATAL_ERROR(
            "Expected non-null argument. Possible cause: proteus::jit_array "
            "argument is not an argument of the enclosing function.");

      auto &ConstantArgs = RCInfoMap[JitFunction];
      RuntimeConstantType RCType = RuntimeConstantType::ARRAY;
      int32_t ArgNo = Arg->getArgNo();

      int32_t NumElts = 0;
      RuntimeConstantType EltType =
          convertTypeToRuntimeConstantType(Elt->getType());
      auto *ConstInt = dyn_cast<ConstantInt>(Size);
      if (ConstInt) {
        NumElts = ConstInt->getZExtValue();

        RuntimeConstantInfo RCI{RCType, ArgNo, NumElts, EltType};
        if (!ConstantArgs.insert(RCI))
          PROTEUS_FATAL_ERROR("Duplicate argument number found: " +
                              std::to_string(Arg->getArgNo()));
      } else {
        // Find NumElts corresponding to a runtime constant argument.
        Argument *NumEltsArg = getOriginatingArgument(Size);
        if (!NumEltsArg)
          PROTEUS_FATAL_ERROR("Expected non-null argument for size");

        RuntimeConstantType NumEltsType =
            convertTypeToRuntimeConstantType(NumEltsArg->getType());
        int32_t NumEltsPos = NumEltsArg->getArgNo();

        RuntimeConstantInfo RCI{RCType, ArgNo, EltType, NumEltsType,
                                NumEltsPos};
        if (!ConstantArgs.insert(RCI))
          PROTEUS_FATAL_ERROR("Duplicate argument number found: " +
                              std::to_string(Arg->getArgNo()));
      }
    }
  }

  // Remove JitArray annotations and their calls as they are not needed anymore.
  // Also removing them avoids errors with LTO and O0 compilation: LTO will
  // find those annotations and attempt to create a manifest which will cause
  // an error since it is not backed by a source filename.
  for (CallBase *CB : CallsToDelete)
    CB->eraseFromParent();

  for (Function *AnnotationF : JitArrayAnnotations)
    AnnotationF->eraseFromParent();

  if (verifyModule(M, &errs()))
    PROTEUS_FATAL_ERROR("Broken JIT module found, compilation aborted!");
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

    DEBUG(Logger::logs("proteus-pass")
          << "JIT Function " << Fn->getName() << "\n");

    auto *Annotation =
        cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));

    DEBUG(Logger::logs("proteus-pass")
          << "Annotation " << Annotation->getAsCString() << "\n");

    // Needs CString for comparison to strip extract null byte character.
    if (Annotation->getAsCString().compare("jit"))
      continue;

    // Get or create the JitFunctionInfo for Fn.
    JitFunctionInfo &JFI = JitFunctionInfoMap[Fn];

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

    SmallSetVector<RuntimeConstantInfo, 16> ParsedRuntimeConstantInfo;
    for (unsigned int J = 0; J < AnnotArgs->getNumOperands(); ++J) {
      std::optional<uint64_t> OptArgNo;
      std::optional<RuntimeConstantType> OptRCType;
      std::optional<int32_t> OptNumElts = std::nullopt;
      std::optional<RuntimeConstantType> OptEltType = std::nullopt;
      std::optional<int32_t> OptNumEltsPos = std::nullopt;
      std::optional<RuntimeConstantType> OptNumEltsType = std::nullopt;

      Constant *C = AnnotArgs->getOperand(J)->stripPointerCasts();
      // We parse either numeric arguments in short form or the key-value pairs
      // in long form.
      if (isa<ConstantInt>(C)) {
        auto *Index = cast<ConstantInt>(AnnotArgs->getOperand(J));
        OptArgNo = Index->getValue().getZExtValue();
        if (*OptArgNo > Fn->arg_size())
          PROTEUS_FATAL_ERROR("Expected ArgNo (1-index) " +
                              std::to_string(*OptArgNo) + " <= arg_size " +
                              std::to_string(Fn->arg_size()) +
                              " for function " + Fn->getName());
        int32_t ArgNo = *OptArgNo - 1;
        OptRCType =
            convertTypeToRuntimeConstantType(Fn->getArg(ArgNo)->getType());
        OptNumElts = 1;
        OptEltType = RuntimeConstantType::NONE;
      } else {
        auto *CDA = dyn_cast<ConstantDataArray>(C->getOperand(0));
        if (!CDA)
          PROTEUS_FATAL_ERROR("Expected constant data array string");

        StringRef RCDictStr = CDA->getAsCString();
        auto ExpectedRCDict = json::parse(RCDictStr);
        if (auto E = ExpectedRCDict.takeError())
          PROTEUS_FATAL_ERROR(
              "Failed to parse runtime constant info json dict");

        json::Value RCDictV = *ExpectedRCDict;
        json::Object *RCDict = RCDictV.getAsObject();
        for (auto &[K, V] : *RCDict) {
          auto OptIntegerValue = V.getAsInteger();
          if (!OptIntegerValue) {
            std::string ErrMsg;
            raw_string_ostream OS{ErrMsg};
            OS << "Expected integer value for key " << K << " value: " << V;
            PROTEUS_FATAL_ERROR(ErrMsg);
          }

          int64_t IntegerValue = *OptIntegerValue;

          if (K == "arg") {
            OptArgNo = IntegerValue;
            if (*OptArgNo > Fn->arg_size())
              PROTEUS_FATAL_ERROR("Expected ArgNo " +
                                  std::to_string(*OptArgNo) + " <= arg_size " +
                                  std::to_string(Fn->arg_size()) +
                                  " for function " + Fn->getName());
          } else if (K == "type") {
            OptRCType = static_cast<RuntimeConstantType>(IntegerValue);
          } else if (K == "num-elts") {
            OptNumElts = IntegerValue;
          } else if (K == "num-elts-pos") {
            OptNumEltsPos = IntegerValue;
          } else if (K == "num-elts-type") {
            OptNumEltsType = static_cast<RuntimeConstantType>(IntegerValue);
          } else if (K == "elt-type") {
            OptEltType = static_cast<RuntimeConstantType>(IntegerValue);
          } else
            PROTEUS_FATAL_ERROR("Unsupported key: " + K.str());
        }
      }

      // Check for semantic errors.
      if (!OptArgNo)
        PROTEUS_FATAL_ERROR("Missing required argument number");
      // We subtract 1 to convert to 0-start index.
      int32_t ArgNo = *OptArgNo - 1;

      if (!OptRCType) {
        // Auto-detect type for a scalar argument.
        OptRCType =
            convertTypeToRuntimeConstantType(Fn->getArg(ArgNo)->getType());
        if (OptRCType == RuntimeConstantType::PTR)
          PROTEUS_FATAL_ERROR("Auto-detecting argument type requires a scalar "
                              "argument, got PTR instead");
      }

      if (RuntimeConstantType::ARRAY == *OptRCType) {
        if (OptNumElts && OptNumEltsPos)
          PROTEUS_FATAL_ERROR(
              "Setting both NumElts and NumEltsPos is incorrect");

        if (!OptNumElts && !OptNumEltsPos)
          PROTEUS_FATAL_ERROR("Either NumElts or NumEltsPos must be set");

        if (OptNumEltsPos && !OptNumEltsType)
          PROTEUS_FATAL_ERROR("NumEltsType must be set if NumEltsPos is set");

        if (!OptNumEltsPos && OptNumEltsType)
          PROTEUS_FATAL_ERROR("NumEltsPos must be set if NumEltsType is set");
      }

      auto CreateRuntimeConstantInfo = [&]() -> RuntimeConstantInfo {
        if (RuntimeConstantType::ARRAY == *OptRCType) {
          if (OptNumElts)
            return RuntimeConstantInfo{*OptRCType, ArgNo, *OptNumElts,
                                       *OptEltType};

          if (OptNumEltsPos && OptNumEltsType)
            return RuntimeConstantInfo{*OptRCType, ArgNo, *OptEltType,
                                       *OptNumEltsType, *OptNumEltsPos};

          PROTEUS_FATAL_ERROR(
              "Unreachable, error in ARRAY runtime constant info definition");
        }

        return RuntimeConstantInfo{*OptRCType, ArgNo};
      };

      RuntimeConstantInfo RCI = CreateRuntimeConstantInfo();
      if (!ParsedRuntimeConstantInfo.insert(RCI))
        PROTEUS_FATAL_ERROR(
            "Duplicate JIT annotation for argument (0-index): " +
            std::to_string(ArgNo));
    }

    // Insert RC infos and sort JFI.ConstantArgs for determinism.
    for (auto &RCInfo : ParsedRuntimeConstantInfo) {
      if (!JFI.ConstantArgs.insert(RCInfo)) {
        PROTEUS_FATAL_ERROR("Duplicate entry found for arg no " +
                            std::to_string(RCInfo.ArgInfo.Pos));
      }
    }
    SmallVector<RuntimeConstantInfo> SortedRCInfos{JFI.ConstantArgs.begin(),
                                                   JFI.ConstantArgs.end()};
    std::sort(SortedRCInfos.begin(), SortedRCInfos.end());
    JFI.ConstantArgs = {SortedRCInfos.begin(), SortedRCInfos.end()};
  }
}

} // namespace proteus
