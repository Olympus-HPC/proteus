//===-- ProteusJitPass.cpp -- Extact code/runtime info for Proteus JIT --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESCRIPTION:
//    Find functions annotated with "jit" plus input arguments that are
//    amenable to runtime constant propagation. Stores the IR for those
//    functions, replaces them with a stub function that calls the jit runtime
//    library to compile the IR and call the function pointer of the jit'ed
//    version.
//
// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libProteusJitPass.dylib -legacy-jit-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libProteusJitPass.dylib -passes="jit-pass" `\`
//        -disable-output <input-llvm-file>
//
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Object/ELF.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FileSystem/UniqueID.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/IPO/MergeFunctions.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/IPO/StripSymbols.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <iostream>
#include <string>
#include <variant>

#include "proteus/Cloning.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/Hashing.hpp"
#include "proteus/Logger.hpp"

#define DEBUG_TYPE "jitpass"
#ifdef PROTEUS_ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

#if PROTEUS_ENABLE_HIP
constexpr char const *RegisterFunctionName = "__hipRegisterFunction";
constexpr char const *LaunchFunctionName = "hipLaunchKernel";
constexpr char const *RegisterVarName = "__hipRegisterVar";
constexpr char const *RegisterFatBinaryName = "__hipRegisterFatBinary";
#elif PROTEUS_ENABLE_CUDA
constexpr char const *RegisterFunctionName = "__cudaRegisterFunction";
constexpr char const *LaunchFunctionName = "cudaLaunchKernel";
constexpr char const *RegisterVarName = "__cudaRegisterVar";
constexpr char const *RegisterFatBinaryName = "__cudaRegisterFatBinary";
#else
constexpr char const *RegisterFunctionName = nullptr;
constexpr char const *LaunchFunctionName = nullptr;
constexpr char const *RegisterVarName = nullptr;
constexpr char const *RegisterFatBinaryName = nullptr;
#endif

using namespace llvm;
using namespace proteus;

//-----------------------------------------------------------------------------
// ProteusJitPass implementation
//-----------------------------------------------------------------------------
namespace {

class ProteusJitPassImpl {
public:
  ProteusJitPassImpl(Module &M) {
    PtrTy = PointerType::getUnqual(M.getContext());
    VoidTy = Type::getVoidTy(M.getContext());
    Int8Ty = Type::getInt8Ty(M.getContext());
    Int32Ty = Type::getInt32Ty(M.getContext());
    Int64Ty = Type::getInt64Ty(M.getContext());
    Int128Ty = Type::getInt128Ty(M.getContext());
    // llvm.global.annotations entry format:
    //  ptr: (addrspace 1) Function pointer
    //  ptr: (addrspace 4) Annotations string
    //  ptr: (addrspace 4) Source file string
    //  i32: Line number,
    //  ptr: (addrspace 1) Arguments pointer
    if (isDeviceCompilation(M)) {
      constexpr unsigned GlobalAddressSpace = 1;
      constexpr unsigned ConstAddressSpace = 4;
      GlobalAnnotationEltTy = StructType::get(
          PointerType::get(M.getContext(), GlobalAddressSpace),
          PointerType::get(M.getContext(), ConstAddressSpace),
          PointerType::get(M.getContext(), ConstAddressSpace), Int32Ty,
          PointerType::get(M.getContext(), GlobalAddressSpace));
    } else
      GlobalAnnotationEltTy =
          StructType::get(PtrTy, PtrTy, PtrTy, Int32Ty, PtrTy);
  }

  bool run(Module &M, bool IsLTO) {
    parseAnnotations(M);

    DEBUG(Logger::logs("proteus-pass")
          << "=== Pre Original Host Module\n"
          << M << "=== End of Pre Original Host Module\n");

    // ==================
    // Device compilation
    // ==================

    // For device compilation, just extract the module IR of device code
    // and return.
    if (isDeviceCompilation(M)) {
      emitJitModuleDevice(M, IsLTO);

      return true;
    }

    // ================
    // Host compilation
    // ================

    instrumentRegisterLinkedBinary(M);
    instrumentRegisterFatBinary(M);
    instrumentRegisterFatBinaryEnd(M);
    instrumentRegisterVar(M);
    findJitVariables(M);
    registerLambdaFunctions(M);

    if (hasDeviceLaunchKernelCalls(M)) {
      getKernelHostStubs(M);
      parseManifestFileAnnotations(M);
      instrumentRegisterFunction(M);
      emitJitLaunchKernelCall(M);
    }

    for (auto &JFI : JitFunctionInfoMap) {
      Function *JITFn = JFI.first;
      DEBUG(Logger::logs("proteus-pass")
            << "Processing JIT Function " << JITFn->getName() << "\n");
      // Skip host device stubs coming from kernel annotations.
      if (isDeviceKernelHostStub(*JITFn))
        continue;

      emitJitModuleHost(M, JFI);
      emitJitEntryCall(M, JFI);
    }

    DEBUG(Logger::logs("proteus-pass")
          << "=== Post Original Host Module\n"
          << M << "=== End Post Original Host Module\n");

    if (verifyModule(M, &errs()))
      PROTEUS_FATAL_ERROR("Broken original module found, compilation aborted!");

    return true;
  }

private:
  Type *PtrTy = nullptr;
  Type *VoidTy = nullptr;
  Type *Int8Ty = nullptr;
  Type *Int32Ty = nullptr;
  Type *Int64Ty = nullptr;
  Type *Int128Ty = nullptr;
  StructType *GlobalAnnotationEltTy = nullptr;

  struct JitFunctionInfo {
    SmallSetVector<int, 16> ConstantArgs;
    std::string ModuleIR;
  };

  MapVector<Function *, JitFunctionInfo> JitFunctionInfoMap;
  DenseMap<Value *, GlobalVariable *> StubToKernelMap;
  SmallPtrSet<Function *, 16> ModuleDeviceKernels;

  bool isDeviceCompilation(Module &M) {
    Triple TargetTriple(M.getTargetTriple());
    DEBUG(Logger::logs("proteus-pass")
          << "TargetTriple " << M.getTargetTriple() << "\n");
    if (TargetTriple.isNVPTX() || TargetTriple.isAMDGCN())
      return true;

    return false;
  }

  bool isDeviceKernel(const Function *F) {
    if (ModuleDeviceKernels.contains(F))
      return true;

    return false;
  }

  bool isLambdaFunction(const Function &F) {
    std::string DemangledName = demangle(F.getName().str());
    return StringRef{DemangledName}.contains("'lambda") &&
           StringRef{DemangledName}.contains(")::operator()");
  }

  std::string getUniqueFileID(Module &M) {
    llvm::sys::fs::UniqueID ID;
    if (auto EC = llvm::sys::fs::getUniqueID(M.getSourceFileName(), ID))
      PROTEUS_FATAL_ERROR("Could not get unique id for source file " +
                          EC.message());

    SmallString<64> Out;
    llvm::raw_svector_ostream OutStr(Out);
    OutStr << llvm::format("%x_%x", ID.getDevice(), ID.getFile());

    return std::string(Out);
  }

  void parseAttributeAnnotations(Module &M, GlobalVariable *GlobalAnnotations) {
    auto *Array = cast<ConstantArray>(GlobalAnnotations->getOperand(0));
    DEBUG(Logger::logs("proteus-pass")
          << "Annotation Array " << *Array << "\n");
    for (unsigned int I = 0; I < Array->getNumOperands(); I++) {
      auto *Entry = cast<ConstantStruct>(Array->getOperand(I));
      DEBUG(Logger::logs("proteus-pass") << "Entry " << *Entry << "\n");

      auto *Fn = dyn_cast<Function>(Entry->getOperand(0)->stripPointerCasts());

      assert(Fn && "Expected function in entry operands");

      // Check the annotated functions is a kernel function or a device
      // lambda.
      if (isDeviceCompilation(M)) {
        ModuleDeviceKernels = getDeviceKernels(M);
        if (!isDeviceKernel(Fn) && !isLambdaFunction(*Fn))
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

      // TODO: needs CString for comparison to work, why?
      if (Annotation->getAsCString().compare("jit"))
        continue;

      JitFunctionInfo JFI;

      if (Entry->getOperand(4)->isNullValue())
        JFI.ConstantArgs = {};
      else {
        DEBUG(Logger::logs("proteus-pass")
              << "AnnotArgs " << *Entry->getOperand(4)->getOperand(0) << "\n");
        DEBUG(Logger::logs("proteus-pass")
              << "Type AnnotArgs "
              << *Entry->getOperand(4)->getOperand(0)->getType() << "\n");
        auto *AnnotArgs =
            cast<ConstantStruct>(Entry->getOperand(4)->getOperand(0));

        SmallSetVector<int, 16> JitArgs;
        for (unsigned int J = 0; J < AnnotArgs->getNumOperands(); ++J) {
          auto *Index = cast<ConstantInt>(AnnotArgs->getOperand(J));
          uint64_t ArgNo = Index->getValue().getZExtValue();
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
      }

      JitFunctionInfoMap[Fn] = JFI;
    }
  }

  SmallString<64> getUniqueManifestFilename(Module &M) {
    auto TmpPath = std::filesystem::temp_directory_path();

    return {TmpPath.string(), "/", "proteus-device-manifest-",
            getUniqueFileID(M), ".json"};
  }

  void createDeviceManifestFile(
      Module &M, DenseMap<Function *, SmallSetVector<int, 16>> &JitArgs) {
    // Emit JSON file manifest which contains the kernel symbol and
    // JIT-annotated arguments.
    SmallString<64> UniqueFilename = getUniqueManifestFilename(M);
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

  void appendToGlobalAnnotations(Module &M,
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
        ArrayType::get(GlobalAnnotationEltTy, CurrentAnnotations.size());
    Constant *Init = ConstantArray::get(AT, CurrentAnnotations);
    auto *AnnotationsGV = new GlobalVariable(M, Init->getType(), false,
                                             GlobalValue::AppendingLinkage,
                                             Init, "llvm.global.annotations");
    AnnotationsGV->setSection("llvm.metadata");
  }

  Constant *createJitAnnotation(Module &M, Function *F,
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
    SmallVector<Type *> ArgInfo{ConstantArgs.size(), Int32Ty};
    StructType *ArgEltTy = StructType::get(M.getContext(), ArgInfo);
    SmallVector<Constant *> ArgConsts;
    for (int ArgNo : ConstantArgs)
      // We add 1 to the ArgNo to create the 1-index argument number that global
      // annotations expect.
      // TODO: Maybe require 0-indexing to avoid this?
      ArgConsts.push_back(ConstantInt::get(Int32Ty, ArgNo + 1));
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
        GlobalAnnotationEltTy, ArrayRef{AnnotationVals, NumElts});

    return NewAnnotation;
  }

  /// If V ultimately came from a store of an argument into an alloca, return
  /// that argument, otherwise return nullptr.
  Argument *getOriginatingArgument(Value *V) {
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

    auto *AI = dyn_cast<AllocaInst>(LI->getPointerOperand());
    if (!AI)
      return nullptr;

    for (auto *U : AI->users()) {
      auto *SI = dyn_cast<StoreInst>(U);
      if (!SI || SI->getPointerOperand() != AI)
        continue;

      Value *Val = StripAllCasts(SI->getValueOperand());
      if (auto *Arg = dyn_cast<Argument>(Val))
        return Arg;
    }

    return nullptr;
  }

  void parseJitArgAnnotations(Module &M,
                              SmallPtrSetImpl<Function *> &JitArgAnnotations) {
    // Iterate over all proteus::jit_arg annotations and store the information
    // in the JitArgs map.
    DenseMap<Function *, SmallSetVector<int, 16>> JitArgs;
    for (Function *AnnotationF : JitArgAnnotations) {
      for (User *Usr : AnnotationF->users()) {
        CallBase *CB = dyn_cast<CallBase>(Usr);
        if (!CB)
          continue;

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
    // Sort argument numbers for determinism.
    SmallVector<Constant *> NewJitAnnotations;
    for (auto &[F, ConstantArgs] : JitArgs) {
      SmallVector<int> SortedArgs{ConstantArgs.begin(), ConstantArgs.end()};
      std::sort(SortedArgs.begin(), SortedArgs.end());
      ConstantArgs = {SortedArgs.begin(), SortedArgs.end()};

      NewJitAnnotations.push_back(createJitAnnotation(M, F, SortedArgs));
    }

    // We append to global annotations the parsed information from the manifest
    // file. This is needed for HIP LTO because it uses global annotations to
    // identify kernels.
    appendToGlobalAnnotations(M, NewJitAnnotations);
    // If this is device compilation the pass emits a JSON file that stores this
    // information for the host compilation pass to parse for instrumentation.
    // The JSON file is uniquely named using the TU unique file ID.
    if (isDeviceCompilation(M))
      createDeviceManifestFile(M, JitArgs);
  }

  void parseAnnotations(Module &M) {
    // First parse any proteus::jit_arg annotations and append them to global
    // annotations.
    SmallPtrSet<Function *, 32> JitArgAnnotations;
    for (auto &F : M.getFunctionList()) {
      std::string DemangledName = demangle(F.getName().str());
      if (StringRef{DemangledName}.contains("proteus::jit_arg"))
        JitArgAnnotations.insert(&F);
    }

    if (!JitArgAnnotations.empty())
      parseJitArgAnnotations(M, JitArgAnnotations);

    // Last, parse global annotations, either created throught attributes or the
    // parsed proteus::jit_arg interface.
    auto *GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
    if (!JitArgAnnotations.empty() && !GlobalAnnotations)
      PROTEUS_FATAL_ERROR("Expected llvm.global.annotations global variable "
                          "after proteus::jit_arg annotations are parsed.");

    if (GlobalAnnotations)
      parseAttributeAnnotations(M, GlobalAnnotations);
  }

  void parseManifestFileAnnotations(Module &M) {
    // Parse the JSON manifest from device compilation, if there exists one, and
    // update JitFunctionInfoMap.
    SmallString<64> UniqueFilename = getUniqueManifestFilename(M);
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

  void runCleanupPassPipeline(Module &M) {
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager Passes;
    Passes.addPass(MergeFunctionsPass());
    Passes.addPass(GlobalOptPass());
    Passes.addPass(GlobalDCEPass());
    Passes.addPass(StripDeadDebugInfoPass());
    Passes.addPass(StripDeadPrototypesPass());

    Passes.run(M, MAM);

    StripDebugInfo(M);
  }

  void runOptimizationPassPipeline(Module &M) {
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager Passes =
        PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
    Passes.run(M, MAM);
  }

  RuntimeConstantTypes convertTypeToRuntimeConstantType(Type *Ty) {
    if (Ty->isIntegerTy(1))
      return RuntimeConstantTypes::BOOL;
    if (Ty->isIntegerTy(8))
      return RuntimeConstantTypes::INT8;
    if (Ty->isIntegerTy(32))
      return RuntimeConstantTypes::INT32;
    if (Ty->isIntegerTy(64))
      return RuntimeConstantTypes::INT64;
    if (Ty->isFloatTy())
      return RuntimeConstantTypes::FLOAT;
    if (Ty->isDoubleTy())
      return RuntimeConstantTypes::DOUBLE;
    if (Ty->isFP128Ty() || Ty->isPPC_FP128Ty() || Ty->isX86_FP80Ty())
      return RuntimeConstantTypes::LONG_DOUBLE;
    if (Ty->isPointerTy())
      return RuntimeConstantTypes::PTR;

    std::string TypeString;
    raw_string_ostream TypeOstream(TypeString);
    Ty->print(TypeOstream);
    PROTEUS_FATAL_ERROR("Unknown Type " + TypeOstream.str());
  }

  void emitJitModuleHost(Module &M,
                         std::pair<Function *, JitFunctionInfo> &JITInfo) {
    Function *JITFn = JITInfo.first;
    JitFunctionInfo &JFI = JITInfo.second;

    ValueToValueMapTy VMap;
    auto JitMod = CloneModule(M, VMap, [](const GlobalValue *GV) {
      if (const GlobalVariable *G = dyn_cast<GlobalVariable>(GV))
        if (!G->isConstant())
          return false;

      return true;
    });

    Function *JitF = cast<Function>(VMap[JITFn]);
    JitF->setLinkage(GlobalValue::ExternalLinkage);

    // Internalize functions, besides JIT function, in the module
    // to help global DCE (reduce compilation time), inlining.
    for (Function &JitModF : *JitMod) {
      if (JitModF.isDeclaration())
        continue;

      if (&JitModF == JitF)
        continue;

      // Internalize other functions in the module.
      JitModF.setLinkage(GlobalValue::InternalLinkage);
    }

    DEBUG(Logger::logs("proteus-pass")
          << "=== Pre Passes Host JIT Module\n"
          << *JitMod << "=== End of Pre Passes Host JIT Module\n");

    // Run a global DCE pass and O3 on the JIT module IR to remove unnecessary
    // symbols and reduce the IR to JIT at runtime.
    runCleanupPassPipeline(*JitMod);
    runOptimizationPassPipeline(*JitMod);

    // Update linkage and visibility in the original module only for
    // globals included in the JIT module required for external
    // linking.
    for (auto &GVar : M.globals()) {
      [[maybe_unused]] auto PrintGVarInfo = [](auto &GVar) {
        Logger::logs("proteus-pass") << "=== GVar\n";
        Logger::logs("proteus-pass") << GVar.getName() << "\n";
        Logger::logs("proteus-pass") << "Linkage " << GVar.getLinkage() << "\n";
        Logger::logs("proteus-pass")
            << "Visibility " << GVar.getVisibility() << "\n";
        Logger::logs("proteus-pass") << "=== End GV\n";
      };

      if (VMap[&GVar]) {
        DEBUG(PrintGVarInfo(GVar));

        if (GVar.isConstant())
          continue;

        if (GVar.getName() == "llvm.global_ctors") {
          DEBUG(Logger::logs("proteus-pass") << "Skip llvm.global_ctors");
          continue;
        }

        if (GVar.hasAvailableExternallyLinkage()) {
          DEBUG(Logger::logs("proteus-pass") << "Skip available externally");
          continue;
        }

        GVar.setLinkage(GlobalValue::ExternalLinkage);
        GVar.setVisibility(GlobalValue::VisibilityTypes::DefaultVisibility);
      }
    }

    // TODO: Do we want to keep debug info to use for GDB/LLDB
    // interfaces for debugging jitted code?
    StripDebugInfo(*JitMod);

    // Add metadata for the JIT function to store the argument positions for
    // runtime constants.
    emitJitFunctionArgMetadata(*JitMod, JFI, *JitF);

    if (verifyModule(*JitMod, &errs()))
      PROTEUS_FATAL_ERROR("Broken JIT module found, compilation aborted!");

    raw_string_ostream OS(JFI.ModuleIR);
    WriteBitcodeToFile(*JitMod, OS);
    OS.flush();

    DEBUG(Logger::logs("proteus-pass")
          << "=== Final Host JIT Module\n"
          << *JitMod << "=== End of Final Host JIT Module\n");
  }

  // Returns true for globals that are not needed when cloning to extract a
  // module.
  static bool skipGlobal(const GlobalValue &G) {
    if (G.getName().starts_with("_jit_bitcode") ||
        G.getName().starts_with("__clang_gpu_used_external") ||
        G.getName().starts_with("__hip_cuid") ||
        G.getName().starts_with("llvm.used") ||
        G.getName().starts_with("llvm.compiler.used") ||
        G.getName().starts_with("llvm.global.annotations"))
      return true;

    return false;
  }

  // Emit a uniquely named global variable in a corresponding section that
  // contains the embedded bitcode module.
  void emitModuleDevice(Module &M, Module &EmbedM, StringRef Id,
                        bool HasSourceFileID) {
    SmallVector<char> Bitcode;
    raw_svector_ostream OS(Bitcode);
    WriteBitcodeToFile(EmbedM, OS);

    HashT HashValue = hash(StringRef{Bitcode.data(), Bitcode.size()});

    std::string GVName = "_jit_bitcode_" + Id.str() +
                         (HasSourceFileID ? ("_" + getUniqueFileID(M)) : "");
    //  NOTE: HIP compilation supports custom section in the binary to store
    //  the IR. CUDA does not, hence we parse the IR by reading the global
    //  from the device memory.
    Constant *JitModule = ConstantDataArray::get(
        M.getContext(),
        ArrayRef<uint8_t>((const uint8_t *)Bitcode.data(), Bitcode.size()));
    if (M.getNamedGlobal(GVName))
      PROTEUS_FATAL_ERROR(
          "Expected unique name for jit module global variable " + GVName);
    auto *GV =
        new GlobalVariable(M, JitModule->getType(), /* isConstant */ true,
                           GlobalValue::ExternalLinkage, JitModule, GVName);
    appendToUsed(M, {GV});
    // We append the hash value to the section name and retrieve it in HIP JIT
    // compilation to avoid hashing at runtime.
    GV->setSection(".jit.bitcode." + Id.str() + getUniqueModuleId(&M) + "." +
                   HashValue.toString());
    DEBUG(Logger::logs("proteus-pass")
          << "Emit jit bitcode GV " << GVName << "\n");
  }

  void emitLinkedKernelModules(Module &LTOModule) {
    LLVMContext Ctx;
    auto LinkedModule = std::make_unique<Module>("linked.module", Ctx);
    LinkedModule->setSourceFileName("linked.module");
    LinkedModule->setDataLayout(LTOModule.getDataLayout());
    LinkedModule->setTargetTriple(LTOModule.getTargetTriple());
    LinkedModule->setModuleInlineAsm(LTOModule.getModuleInlineAsm());
#if LLVM_VERSION_MAJOR >= 18
    LinkedModule->IsNewDbgInfoFormat = LTOModule.IsNewDbgInfoFormat;
#endif

    // Gather all extracted modules and the pruned LTO module.
    SmallVector<std::unique_ptr<Module>> LinkedModules;
    StringSet KernelSymbols;

    auto ExtractKernelSymbols = [&LTOModule, &KernelSymbols]() {
      auto *GlobalAnnotations =
          LTOModule.getNamedGlobal("llvm.global.annotations");
      if (!GlobalAnnotations)
        return;

      auto *AnnotArray = cast<ConstantArray>(GlobalAnnotations->getOperand(0));

      for (unsigned int I = 0; I < AnnotArray->getNumOperands(); I++) {
        auto *Entry = cast<ConstantStruct>(AnnotArray->getOperand(I));
        auto *Fn =
            dyn_cast<Function>(Entry->getOperand(0)->stripPointerCasts());
        assert(Fn && "Expected function in entry operands");
        KernelSymbols.insert(Fn->getName());
      }
    };

    ExtractKernelSymbols();

    SmallPtrSet<GlobalVariable *, 32> RemoveModuleGVs;
    auto ExtractLinkedModules = [&Ctx, &LTOModule, &LinkedModules,
                                 &RemoveModuleGVs]() {
      StringSet DefSet;
      for (auto &GVar : LTOModule.globals()) {
        if (!GVar.hasName())
          continue;

        if (!GVar.getName().starts_with("_jit_bitcode"))
          continue;

        // Found an extracted bitcode global variable, parse the IR and
        // store the names of definitions to prune the LTO module.
        ConstantDataArray *JitModule =
            dyn_cast<ConstantDataArray>(GVar.getInitializer());
        assert(JitModule && "Expected non-null bitcode for JIT module");
        StringRef BitcodeData = JitModule->getAsString();

        // It is important to preserve the unique global variable name as
        // the module name for the parsed module. It will be used later to
        // create unique symbols for linking modules together.
        MemoryBufferRef MBRef{BitcodeData, GVar.getName()};
        auto ExpectedParsedModule = parseBitcodeFile(MBRef, Ctx);
        if (auto E = ExpectedParsedModule.takeError())
          PROTEUS_FATAL_ERROR("Error: " + toString(std::move(E)));
        auto ParsedModule = std::move(*ExpectedParsedModule);
        for (auto &G : ParsedModule->global_values()) {
          if (!G.isDeclaration())
            DefSet.insert(G.getName());
        }

        LinkedModules.push_back(std::move(ParsedModule));

        RemoveModuleGVs.insert(&GVar);
      }

      auto ShouldClone = [&DefSet](const GlobalValue *GV) {
        if (GV->hasName()) {
          if (DefSet.contains(GV->getName()))
            return false;

          if (skipGlobal(*GV))
            return false;
        }

        return true;
      };
      // Create pruned LTO module avoiding unneeded globals and globals in
      // the DefSet. Roundtrip it through bitcode parsing to the top-level
      // context.
      ValueToValueMapTy VMap;
      auto PrunedLTOModule = CloneModule(LTOModule, VMap, ShouldClone);
      for (auto &G : PrunedLTOModule->global_values())
        if (G.hasInternalLinkage())
          G.setLinkage(GlobalValue::ExternalLinkage);

      StripDebugInfo(*PrunedLTOModule);

      if (verifyModule(*PrunedLTOModule, &errs()))
        PROTEUS_FATAL_ERROR(
            "Broken pruned lto module found, compilation aborted!");

      SmallVector<char> Bitcode;
      raw_svector_ostream BOS{Bitcode};
      WriteBitcodeToFile(*PrunedLTOModule, BOS);
      MemoryBufferRef MBRef{StringRef{Bitcode.data(), Bitcode.size()},
                            "pruned.lto.module"};
      auto ExpectedPrunedLTOModuleInCtx = parseBitcodeFile(MBRef, Ctx);
      if (auto E = ExpectedPrunedLTOModuleInCtx.takeError())
        PROTEUS_FATAL_ERROR("Error parsing pruned lto module " +
                            toString(std::move(E)));
      LinkedModules.push_back(std::move(*ExpectedPrunedLTOModuleInCtx));
    };

    ExtractLinkedModules();
    // Remove global vars with per-TU modules as they are not needed anymore.
    // The final step in this method will create per-kernel modules.
    for (auto *GV : RemoveModuleGVs) {
      removeFromUsedLists(LTOModule, [GV](Constant *C) {
        if (GV == C)
          return true;

        return false;
      });
      LTOModule.eraseGlobalVariable(GV);
    }

    Linker IRLinker(*LinkedModule);
    for (auto &Mod : LinkedModules) {
      if (IRLinker.linkInModule(std::move(Mod)))
        PROTEUS_FATAL_ERROR("Linking failed");
    }

    runCleanupPassPipeline(*LinkedModule);

    const char *EnvValue = std::getenv("PROTEUS_PASS_CREATE_KERNEL_MODULES");
    bool CreateKernelModules =
        (EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : true);

    if (CreateKernelModules) {
      for (auto &Sym : KernelSymbols) {
        auto KernelName = Sym.getKey();

        if (!LinkedModule->getFunction(KernelName))
          PROTEUS_FATAL_ERROR("Expected kernel function in linked module");

        auto KernelModule = cloneKernelFromModules({*LinkedModule}, KernelName);
        runCleanupPassPipeline(*KernelModule);

        if (verifyModule(*KernelModule, &errs()))
          PROTEUS_FATAL_ERROR(
              "Broken original module found, compilation aborted!");

        emitModuleDevice(LTOModule, *KernelModule, KernelName, false);
      }
    } else {
      emitModuleDevice(LTOModule, *LinkedModule, "lto", false);
    }
  }

  void emitJitModuleDevice(Module &M, bool IsLTO) {
    SmallVector<char> Bitcode;
    raw_svector_ostream OS(Bitcode);

    // For LTO, supported only in HIP RDC, create the per-kernel linked module
    // AOT.
    if (IsLTO) {
      emitLinkedKernelModules(M);
    } else {
      // Emit the TU-wide extracted module.
      ValueToValueMapTy VMap;
      auto ShouldClone = [](const GlobalValue *G) {
        if (skipGlobal(*G))
          return false;

        return true;
      };
      auto EmitM = CloneModule(M, VMap, ShouldClone);
      runCleanupPassPipeline(*EmitM);

      emitModuleDevice(M, *EmitM, "tu", true);
    }
  }

  void emitJitFunctionArgMetadata(Module &JitMod, JitFunctionInfo &JFI,
                                  Function &JitF) {
    LLVMContext &Ctx = JitMod.getContext();
    SmallVector<Metadata *> ConstArgNos;
    for (int ArgNo : JFI.ConstantArgs) {
      Metadata *Meta =
          ConstantAsMetadata::get(ConstantInt::get(Int32Ty, ArgNo));
      ConstArgNos.push_back(Meta);
    }
    MDNode *Node = MDNode::get(Ctx, ConstArgNos);
    JitF.setMetadata("jit_arg_nos", Node);
  }

  FunctionCallee getJitEntryFn(Module &M) {
    // The prototype is
    // __jit_entry(char *FnName,
    //             char *IR,
    //             int IRSize,
    //             void **Args,
    //             int32_t *RCIndices,
    //             int32_t *RCTypes,
    //             int32_t NumRCs)

    FunctionType *JitEntryFnTy = FunctionType::get(
        PtrTy, {PtrTy, PtrTy, Int32Ty, PtrTy, PtrTy, PtrTy, Int32Ty},
        /* isVarArg=*/false);
    FunctionCallee JitEntryFn =
        M.getOrInsertFunction("__jit_entry", JitEntryFnTy);

    return JitEntryFn;
  }

  void emitJitEntryCall(Module &M,
                        std::pair<Function *, JitFunctionInfo> &JITInfo) {

    Function *JITFn = JITInfo.first;
    JitFunctionInfo &JFI = JITInfo.second;
    size_t NumRuntimeConstants = JFI.ConstantArgs.size();

    FunctionCallee JitEntryFn = getJitEntryFn(M);

    // Replaces jit'ed functions in the original module with stubs to call the
    // runtime entry point that compiles and links.
    // Replace jit'ed function with a stub function.
    std::string FnName = JITFn->getName().str();
    JITFn->setName("");
    Function *StubFn = Function::Create(JITFn->getFunctionType(),
                                        JITFn->getLinkage(), FnName, M);
    JITFn->replaceAllUsesWith(StubFn);
    JITFn->eraseFromParent();

    // Replace the body of the jit'ed function to call the jit entry, grab the
    // address of the specialized jit version and execute it.
    IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", StubFn));

    // Create the runtime constant array type for the indices and types of
    // runtime constant info passed to the jit entry function.
    ArrayType *RuntimeConstantArrayInfoTy =
        ArrayType::get(Int32Ty, NumRuntimeConstants);

    SmallVector<Constant *> RCIndices;
    SmallVector<Constant *> RCTypes;
    for (int ArgNo : JFI.ConstantArgs) {
      Constant *ArgNoConstant = ConstantInt::get(Int32Ty, ArgNo);
      RCIndices.push_back(ArgNoConstant);

      int32_t TypeId =
          convertTypeToRuntimeConstantType(StubFn->getArg(ArgNo)->getType());
      Constant *TypeIdConstant = ConstantInt::get(Int32Ty, TypeId);
      RCTypes.push_back(TypeIdConstant);
    }
    Constant *RCIndicesConstant =
        ConstantArray::get(RuntimeConstantArrayInfoTy, RCIndices);
    Constant *RCTypesConstant =
        ConstantArray::get(RuntimeConstantArrayInfoTy, RCTypes);

    // Create globals for the function name, string, RC indices and types.
    // entry.
    auto *FnNameGlobal = Builder.CreateGlobalString(StubFn->getName());
    auto *StrIRGlobal = Builder.CreateGlobalString(JFI.ModuleIR);
    auto *RCIndicesGV = new GlobalVariable(
        M, RuntimeConstantArrayInfoTy, /*isConstant=*/true,
        GlobalVariable::PrivateLinkage,
        /*Initializer=*/RCIndicesConstant, ".proteus.indices");
    auto *RCTypesGV =
        new GlobalVariable(M, RuntimeConstantArrayInfoTy, /*isConstant=*/true,
                           GlobalVariable::PrivateLinkage,
                           /*Initializer=*/RCTypesConstant, ".proteus.types");

    // Create the runtime constants args pointer array.
    ArrayType *ArgPtrsTy = ArrayType::get(PtrTy, StubFn->arg_size());
    Value *ArgPtrs = nullptr;
    if (NumRuntimeConstants > 0) {
      ArgPtrs = Builder.CreateAlloca(ArgPtrsTy);
      // Create an alloca for each argument.
      SmallVector<AllocaInst *> ArgPtrAllocas;
      for (size_t ArgI = 0; ArgI < StubFn->arg_size(); ++ArgI) {
        auto *Alloca = Builder.CreateAlloca(StubFn->getArg(ArgI)->getType());
        ArgPtrAllocas.push_back(Alloca);
      }
      // Store each a pointer to the argument value to each alloca.
      for (size_t ArgI = 0; ArgI < StubFn->arg_size(); ++ArgI) {
        auto *GEP = Builder.CreateInBoundsGEP(
            ArgPtrsTy, ArgPtrs, {Builder.getInt32(0), Builder.getInt32(ArgI)});
        Builder.CreateStore(StubFn->getArg(ArgI), ArgPtrAllocas[ArgI]);
        Builder.CreateStore(ArgPtrAllocas[ArgI], GEP);
      }
    } else
      ArgPtrs = Constant::getNullValue(ArgPtrsTy->getPointerTo());

    auto *JitFnPtr =
        Builder.CreateCall(JitEntryFn, {FnNameGlobal, StrIRGlobal,
                                        Builder.getInt32(JFI.ModuleIR.size()),
                                        ArgPtrs, RCIndicesGV, RCTypesGV,
                                        Builder.getInt32(NumRuntimeConstants)});
    SmallVector<Value *, 8> Args;
    for (auto &Arg : StubFn->args())
      Args.push_back(&Arg);
    auto *RetVal =
        Builder.CreateCall(StubFn->getFunctionType(), JitFnPtr, Args);
    if (StubFn->getReturnType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(RetVal);
  }

  Value *getStubGV([[maybe_unused]] Value *Operand) {
    // NOTE: when called by isDeviceKernelHostStub, Operand may not be a global
    // variable point to the stub, so we check and return null in that case.
    Value *V = nullptr;
#if PROTEUS_ENABLE_HIP
    // NOTE: Hip creates a global named after the device kernel function that
    // points to the host kernel stub. Because of this, we need to unpeel this
    // indirection to use the host kernel stub for finding the device kernel
    // function name global.
    GlobalVariable *IndirectGV = dyn_cast<GlobalVariable>(Operand);
    V = IndirectGV ? IndirectGV->getInitializer() : nullptr;
#elif PROTEUS_ENABLE_CUDA
    GlobalValue *DirectGV = dyn_cast<GlobalValue>(Operand);
    V = DirectGV ? DirectGV : nullptr;
#endif

    return V;
  }

  void getKernelHostStubs(Module &M) {
    Function *RegisterFunction = nullptr;
    if (!RegisterFunctionName) {
      PROTEUS_FATAL_ERROR("getKernelHostStubs only callable with `EnableHIP or "
                          "EnableCUDA set.");
      return;
    }
    RegisterFunction = M.getFunction(RegisterFunctionName);

    if (!RegisterFunction)
      return;

    constexpr int StubOperand = 1;
    constexpr int KernelOperand = 2;
    for (User *Usr : RegisterFunction->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        GlobalVariable *GV =
            dyn_cast<GlobalVariable>(CB->getArgOperand(KernelOperand));
        assert(GV && "Expected global variable as kernel name operand");
        Value *Key = getStubGV(CB->getArgOperand(StubOperand));
        assert(Key && "Expected valid kernel stub key");
        StubToKernelMap[Key] = GV;
        DEBUG(Logger::logs("proteus-pass")
              << "StubToKernelMap Key: " << Key->getName() << " -> " << *GV
              << "\n");
      }
  }

  SmallPtrSet<Function *, 16> getDeviceKernels([[maybe_unused]] Module &M) {
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
  }

  bool isDeviceKernelHostStub(Function &Fn) {
    if (StubToKernelMap.contains(&Fn))
      return true;

    return false;
  }

  bool hasDeviceLaunchKernelCalls(Module &M) {
    Function *LaunchKernelFn = nullptr;
    if (!LaunchFunctionName) {
      return false;
    }
    LaunchKernelFn = M.getFunction(LaunchFunctionName);

    if (!LaunchKernelFn)
      return false;

    return true;
  }

  FunctionCallee getJitLaunchKernelFn(Module &M) {
    FunctionType *JitLaunchKernelFnTy = nullptr;

    assert(LaunchFunctionName && "Expected valid launch function name");
    Function *LaunchKernelFn = M.getFunction(LaunchFunctionName);
    assert(LaunchKernelFn && "Expected non-null launch kernel function");

    // The ABI of __jit_launch_kernel mirrors the device-specific
    // launchKernel. Note the ABI can be different depending on the host
    // architecture.
    JitLaunchKernelFnTy = LaunchKernelFn->getFunctionType();

    if (!JitLaunchKernelFnTy)
      PROTEUS_FATAL_ERROR(
          "Expected non-null jit entry function type, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusJitPass");

    FunctionCallee JitLaunchKernelFn =
        M.getOrInsertFunction("__jit_launch_kernel", JitLaunchKernelFnTy);

    return JitLaunchKernelFn;
  }

  void replaceWithJitLaunchKernel(Module &M, CallBase *LaunchKernelCB) {
    FunctionCallee JitLaunchKernelFn = getJitLaunchKernelFn(M);

    // Insert before the launch kernel call instruction.
    IRBuilder<> Builder(LaunchKernelCB);
    CallBase *CallOrInvoke = nullptr;

    SmallVector<Value *> Args = {LaunchKernelCB->arg_begin(),
                                 LaunchKernelCB->arg_end()};

    if (auto *CallI = dyn_cast<CallInst>(LaunchKernelCB)) {
      CallOrInvoke = Builder.CreateCall(JitLaunchKernelFn, Args);
    } else if (auto *InvokeI = dyn_cast<InvokeInst>(LaunchKernelCB)) {
      CallOrInvoke =
          Builder.CreateInvoke(JitLaunchKernelFn, InvokeI->getNormalDest(),
                               InvokeI->getUnwindDest(), Args);
    }

    if (!CallOrInvoke)
      PROTEUS_FATAL_ERROR(
          "Expected non-null jit launch kernel call or invoke, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusJitPass");

    LaunchKernelCB->replaceAllUsesWith(CallOrInvoke);
    LaunchKernelCB->eraseFromParent();
  }

  void emitJitLaunchKernelCall(Module &M) {
    Function *LaunchKernelFn = nullptr;
    if (!LaunchFunctionName) {
      PROTEUS_FATAL_ERROR(
          "Expected non-null LaunchKernelFn, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusJitPass");
    }
    LaunchKernelFn = M.getFunction(LaunchFunctionName);
    if (!LaunchKernelFn)
      PROTEUS_FATAL_ERROR(
          "Expected non-null LaunchKernelFn, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusJitPass");

    SmallVector<CallBase *> ToBeReplaced;
    for (User *Usr : LaunchKernelFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        // NOTE: We search for calls to the LaunchKernelFn that directly call
        // the kernel through its global value to replace with JIT kernel
        // entries. For cudaLaunchKernel first operand is the stub function,
        // whereas for hipLaunchKernel it is a global variable that points to
        // the stub function. Hence we use GlobalValue instead of
        // GlobalVaraible.
        // TODO: Instrument for indirect launching.

        ToBeReplaced.push_back(CB);
      }

    for (CallBase *CB : ToBeReplaced)
      replaceWithJitLaunchKernel(M, CB);
  }

  FunctionCallee getJitRegisterFatBinaryFn(Module &M) {
    FunctionType *JitRegisterFatbinaryFnTy =
        FunctionType::get(VoidTy, {PtrTy, PtrTy, PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisterFatbinaryFn = M.getOrInsertFunction(
        "__jit_register_fatbinary", JitRegisterFatbinaryFnTy);

    return JitRegisterFatbinaryFn;
  }

  void instrumentRegisterFatBinary(Module &M) {
    Function *F = nullptr;

    if (!RegisterFatBinaryName)
      return;

    F = M.getFunction(RegisterFatBinaryName);
    if (!F)
      return;

    FunctionCallee JitRegisterFatBinaryFn = getJitRegisterFatBinaryFn(M);

    for (auto *User : F->users()) {
      CallBase *CB = dyn_cast<CallBase>(User);
      if (!CB)
        continue;

      IRBuilder<> Builder(CB->getNextNode());
      Value *FatbinWrapper = CB->getArgOperand(0);

      std::string GVName = "_jit_bitcode_tu_" + getUniqueFileID(M);
      DEBUG(Logger::logs("proteus-pass")
                << "Instrument register fatbinary bitcode GV " << GVName
                << "\n";);
      auto *Arg = Builder.CreateGlobalString(GVName);

      Builder.CreateCall(JitRegisterFatBinaryFn, {CB, FatbinWrapper, Arg});
    }
  }

  FunctionCallee getJitRegisterFatBinaryEndFn(Module &M) {
    FunctionType *JitRegisterFatBinaryEndFnTy =
        FunctionType::get(VoidTy, {PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisterFatBinaryEndFn = M.getOrInsertFunction(
        "__jit_register_fatbinary_end", JitRegisterFatBinaryEndFnTy);

    return JitRegisterFatBinaryEndFn;
  }

  void instrumentRegisterFatBinaryEnd(Module &M) {
// This is CUDA specific.
#if !PROTEUS_ENABLE_CUDA
    return;
#endif

    Function *F = M.getFunction("__cudaRegisterFatBinaryEnd");
    if (!F)
      return;

    FunctionCallee JitRegisterFatBinaryEndFn = getJitRegisterFatBinaryEndFn(M);

    for (auto *User : F->users()) {
      CallBase *CB = dyn_cast<CallBase>(User);
      if (!CB)
        continue;

      IRBuilder<> Builder(CB->getNextNode());
      Value *FatbinWrapper = CB->getArgOperand(0);
      Builder.CreateCall(JitRegisterFatBinaryEndFn, {FatbinWrapper});
    }
  }

  FunctionCallee getJitRegisterLinkedBinaryFn(Module &M) {
    FunctionType *JitRegisterLinkedBinaryFnTy =
        FunctionType::get(VoidTy, {PtrTy, PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisteLinkedBinaryrFn = M.getOrInsertFunction(
        "__jit_register_linked_binary", JitRegisterLinkedBinaryFnTy);

    return JitRegisteLinkedBinaryrFn;
  }

  void instrumentRegisterLinkedBinary(Module &M) {
// This is CUDA specific.
#if !PROTEUS_ENABLE_CUDA
    return;
#endif

    // Note: we check for __cuda_fatibn_wrapper to avoid emitting for the
    // link.stub. It's not strictly necessary since this module will not have a
    // device bitcode to pull and we skip at runtime.
    if (!M.getGlobalVariable("__cuda_fatbin_wrapper", /*AllowInternal=*/true)) {
      DEBUG(Logger::logs("proteus-pass")
                << "Skip " << M.getSourceFileName() << "\n";)
      return;
    }

    FunctionCallee JitRegisterLinkedBinaryFn = getJitRegisterLinkedBinaryFn(M);

    for (auto &F : M.getFunctionList()) {
      if (!F.getName().starts_with("__cudaRegisterLinkedBinary"))
        continue;

      for (auto *User : F.users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          continue;

        IRBuilder<> Builder(CB);
        std::string GVName = "_jit_bitcode_tu_" + getUniqueFileID(M);
        DEBUG(Logger::logs("proteus-pass")
              << "Instrument register linked binary to extract bitcode GV "
              << GVName << "\n");
        auto *Arg = Builder.CreateGlobalString(GVName);
        Builder.CreateCall(JitRegisterLinkedBinaryFn,
                           {CB->getArgOperand(1), Arg});
      }
    }
  }

  FunctionCallee getJitRegisterVarFn(Module &M) {
    // The prototype is
    // __jit_register_var(const void *HostAddr, const char *VarName).
    FunctionType *JitRegisterVarFnTy = FunctionType::get(PtrTy, {PtrTy, PtrTy},
                                                         /* isVarArg=*/false);
    FunctionCallee JitRegisterVarFn =
        M.getOrInsertFunction("__jit_register_var", JitRegisterVarFnTy);

    return JitRegisterVarFn;
  }

  void instrumentRegisterVar(Module &M) {
    Function *RegisterVarFn = nullptr;
    if (!RegisterVarName)
      return;

    RegisterVarFn = M.getFunction(RegisterVarName);
    if (!RegisterVarFn)
      return;

    FunctionCallee JitRegisterVarFn = getJitRegisterVarFn(M);

    for (User *Usr : RegisterVarFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        IRBuilder<> Builder(CB->getNextNode());
        Value *Symbol = CB->getArgOperand(1);
        auto *GV = dyn_cast<GlobalVariable>(Symbol);
        Value *SymbolName = CB->getArgOperand(2);
        Builder.CreateCall(JitRegisterVarFn, {GV, SymbolName});
      }
  }

  FunctionCallee getJitRegisterFunctionFn(Module &M) {
    // The prototype is
    // __jit_register_function(void *Handle,
    //                         void *Kernel,
    //                         char const *KernelName,
    //                         int32_t *RCIndices,
    //                         int32_t *RCTypes,
    //                         int32_t NumRCs)
    FunctionType *JitRegisterFunctionFnTy =
        FunctionType::get(VoidTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, Int32Ty},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisterKernelFn = M.getOrInsertFunction(
        "__jit_register_function", JitRegisterFunctionFnTy);

    return JitRegisterKernelFn;
  }

  /// instrumentRegisterFunction instruments kernel functions following GPU
  /// runtime registration with __jit_register_function.
  void instrumentRegisterFunction(Module &M) {
    if (!RegisterFunctionName) {
      PROTEUS_FATAL_ERROR(
          "instrumentRegisterJITFunc only callable with `EnableHIP or "
          "EnableCUDA set.");
      return;
    }

    Function *RegisterFunction = M.getFunction(RegisterFunctionName);
    assert(RegisterFunction &&
           "Expected register function to be called at least once.");

    for (User *RegisterFunctionUser : RegisterFunction->users()) {
      CallBase *RegisterCB = dyn_cast<CallBase>(RegisterFunctionUser);
      if (!RegisterCB)
        continue;

      Function *FunctionToRegister =
          dyn_cast<Function>(getStubGV(RegisterCB->getArgOperand(1)));
      assert(FunctionToRegister &&
             "Expected function passed to register function call");
      if (!JitFunctionInfoMap.contains(FunctionToRegister)) {
        DEBUG(Logger::logs("proteus-pass") << "Not instrumenting device kernel "
                                           << *FunctionToRegister << "\n");
        continue;
      }

      DEBUG(Logger::logs("proteus-pass")
            << "Instrumenting JIT function " << *FunctionToRegister << "\n");
      const auto &JFI = JitFunctionInfoMap[FunctionToRegister];
      size_t NumRuntimeConstants = JFI.ConstantArgs.size();
      // Create jit entry runtime function.

      // Both RCIndices and RCIndices have the same array type.
      ArrayType *RuntimeConstantArrayTy =
          ArrayType::get(Int32Ty, NumRuntimeConstants);

      IRBuilder<> Builder(RegisterCB->getNextNode());
      // Create an array representing the indices of the args which are runtime
      // constants.
      Value *RuntimeConstantsIndicesAlloca =
          Builder.CreateAlloca(RuntimeConstantArrayTy);
      assert(RuntimeConstantsIndicesAlloca &&
             "Expected non-null runtime constants alloca");
      // Zero-initialize the alloca to avoid stack garbage for caching.
      Builder.CreateStore(Constant::getNullValue(RuntimeConstantArrayTy),
                          RuntimeConstantsIndicesAlloca);

      // Create an array for the types of runtime constant arguments.
      Value *RuntimeConstantsTypesAlloca =
          Builder.CreateAlloca(RuntimeConstantArrayTy);
      assert(RuntimeConstantsTypesAlloca &&
             "Expected non-null runtime constants alloca");
      // Zero-initialize the alloca.
      Builder.CreateStore(Constant::getNullValue(RuntimeConstantArrayTy),
                          RuntimeConstantsTypesAlloca);

      int GEPIdx = 0;
      for (int ArgNo : JFI.ConstantArgs) {
        auto *GEP = Builder.CreateInBoundsGEP(
            RuntimeConstantArrayTy, RuntimeConstantsIndicesAlloca,
            {Builder.getInt32(0), Builder.getInt32(GEPIdx)});
        Value *Idx = ConstantInt::get(Builder.getInt32Ty(), ArgNo);
        Builder.CreateStore(Idx, GEP);

        auto *GEPType = Builder.CreateInBoundsGEP(
            RuntimeConstantArrayTy, RuntimeConstantsTypesAlloca,
            {Builder.getInt32(0), Builder.getInt32(GEPIdx)});
        int32_t TypeId = convertTypeToRuntimeConstantType(
            FunctionToRegister->getArg(ArgNo)->getType());
        Value *TypeVal = ConstantInt::get(Builder.getInt32Ty(), TypeId);
        Builder.CreateStore(TypeVal, GEPType);

        GEPIdx++;
      }
      Value *NumRCsValue =
          ConstantInt::get(Builder.getInt32Ty(), NumRuntimeConstants);

      FunctionCallee JitRegisterFunction = getJitRegisterFunctionFn(M);

      Builder.CreateCall(
          JitRegisterFunction,
          {RegisterCB->getArgOperand(0), RegisterCB->getArgOperand(1),
           RegisterCB->getArgOperand(2), RuntimeConstantsIndicesAlloca,
           RuntimeConstantsTypesAlloca, NumRCsValue});
    }
  }

  void findJitVariables(Module &M) {
    DEBUG(Logger::logs("proteus-pass") << "finding jit variables" << "\n");
    DEBUG(Logger::logs("proteus-pass") << "users..." << "\n");

    SmallVector<Function *, 16> JitFunctions;

    for (auto &F : M.getFunctionList()) {
      // TODO: Demangle and search for the fully qualified proteus::jit_variable
      // name.
      if (F.getName().contains("jit_variable")) {
        JitFunctions.push_back(&F);
      }
    }

    auto FindStorePtr = [&](CallBase *CB) {
      // Find the store instruction user of the JitVariableCB to extract the
      // pointer to the lambda anonymous class object.
      Value *Ptr = nullptr;
      Value *V = CB;
      while (!Ptr) {
        if (!V->hasOneUser())
          PROTEUS_FATAL_ERROR("Expected single user");

        StoreInst *S = dyn_cast<StoreInst>(*(V->users().begin()));
        if (S) {
          DEBUG(Logger::logs("proteus-pass") << "store: " << *S << "\n");
          Ptr = S->getPointerOperand();
          break;
        }

        // Recurse to the next user.
        V = *V->users().begin();
      }

      return Ptr;
    };

    for (auto *Function : JitFunctions) {
      for (auto *User : Function->users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          PROTEUS_FATAL_ERROR(
              "Expected CallBase as user of proteus::jit_variable function");

        DEBUG(Logger::logs("proteus-pass") << "call: " << *CB << "\n");

        Value *V = FindStorePtr(CB);

        GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V);
        if (GEP) {
          DEBUG(Logger::logs("proteus-pass") << "gep: " << *GEP << "\n");
          auto *Slot = GEP->getOperand(GEP->getNumOperands() - 1);
          DEBUG(Logger::logs("proteus-pass") << "slot: " << *Slot << "\n");
          CB->setArgOperand(1, Slot);
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "no gep, assuming slot 0" << "\n");
          Constant *C = ConstantInt::get(Int32Ty, 0);
          CB->setArgOperand(1, C);
        }
      }
    }
  }

  StringRef parseLambdaType(StringRef DemangledName) {
    int L = -1;
    int R = -1;
    int Level = 0;
    // Start after the function symbol to avoid parsing its templated return
    // type.
    size_t Start = DemangledName.find("proteus::register_lambda");
    for (size_t I = Start, E = DemangledName.size(); I < E; ++I) {
      const char C = DemangledName[I];
      if (C == '<') {
        Level++;
        if (Level == 1)
          L = I;
      }

      if (C == '>') {
        if (Level == 1) {
          R = I;
          break;
        }
        Level--;
      }
    }

    assert(L > 0 && R > L && "Expected non-zero L, R for slicing");
    // Remove reference character '&', if it exists.
    if (DemangledName[R - 1] == '&')
      R--;
    // Slicing returns characters [Start, End).
    return DemangledName.slice(L + 1, R);
    ;
  }

  void registerLambdaFunctions(Module &M) {
    DEBUG(Logger::logs("proteus-pass")
          << "registering lambda functions" << "\n");

    SmallVector<Function *, 16> LambdaFunctions;
    for (auto &F : M.getFunctionList()) {
      if (StringRef{demangle(F.getName().str())}.contains(
              "proteus::register_lambda")) {
        LambdaFunctions.push_back(&F);
      }
    }

    for (auto *Function : LambdaFunctions) {
      auto DemangledName = llvm::demangle(Function->getName().str());
      StringRef LambdaType = parseLambdaType(DemangledName);

      DEBUG(Logger::logs("proteus-pass")
            << Function->getName() << " " << DemangledName << " " << LambdaType
            << "\n");

      for (auto *User : Function->users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          PROTEUS_FATAL_ERROR("Expected CallBase as user of "
                              "proteus::register_lambda function");

        IRBuilder<> Builder(CB);
        auto *LambdaNameGlobal = Builder.CreateGlobalString(LambdaType);
        CB->setArgOperand(1, LambdaNameGlobal);
      }
    }
  }
};

// New PM implementation.
struct ProteusJitPass : PassInfoMixin<ProteusJitPass> {
  ProteusJitPass(bool IsLTO) : IsLTO(IsLTO) {}
  bool IsLTO;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager & /*AM*/) {
    ProteusJitPassImpl PJP{M};

    bool Changed = PJP.run(M, IsLTO);
    if (Changed)
      // TODO: is anything preserved?
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation.
struct LegacyProteusJitPass : public ModulePass {
  static char ID;
  LegacyProteusJitPass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    ProteusJitPassImpl PJP{M};
    bool Changed = PJP.run(M, false);
    return Changed;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getProteusJitPassPluginInfo() {
  const auto Callback = [](PassBuilder &PB) {
    // TODO: decide where to insert it in the pipeline. Early avoids
    // inlining jit function (which disables jit'ing) but may require more
    // optimization, hence overhead, at runtime. We choose after early
    // simplifications which should avoid inlining and present a reasonably
    // analyzable IR module.

    // NOTE: For device jitting it should be possible to register the pass late
    // to reduce compilation time and does lose the kernel due to inlining.
    // However, there are linking errors, working assumption is that the hiprtc
    // linker cannot re-link already linked device libraries and aborts.

    // PB.registerPipelineStartEPCallback(
    // PB.registerOptimizerLastEPCallback(
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(ProteusJitPass{false});
          return true;
        });

    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(ProteusJitPass{true});
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "ProteusJitPass", LLVM_VERSION_STRING,
          Callback};
}

// TODO: use by proteus-jit-pass name.
// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize ProteusJitPass when added to the pass pipeline on the
// command line, i.e. via '-passes=proteus-jit-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getProteusJitPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyProteusJitPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyProteusJitPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-jit-pass'
static RegisterPass<LegacyProteusJitPass>
    X("legacy-jit-pass", "Jit Pass",
      false, // This pass doesn't modify the CFG => false
      false  // This pass is not a pure analysis pass => false
    );
