#ifndef PROTEUS_PASS_HELPERS_H
#define PROTEUS_PASS_HELPERS_H

#include <llvm/ADT/SetVector.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>

#include "proteus/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/Error.h"
#include "proteus/Logger.hpp"

#define DEBUG_TYPE "proteus-pass"
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

namespace proteus {

using namespace llvm;

struct JitFunctionInfo {
  SmallSetVector<RuntimeConstantInfo, 16> ConstantArgs;
  std::string ModuleIR;
};

struct ModuleInfo {
  const Module &M;
  ModuleInfo(const Module &M) : M(M) {}
};

bool inline isDeviceCompilation(Module &M) {
  Triple TargetTriple(M.getTargetTriple());
  DEBUG(Logger::logs("proteus-pass")
        << "TargetTriple " << M.getTargetTriple() << "\n");
  if (TargetTriple.isNVPTX() || TargetTriple.isAMDGCN())
    return true;

  return false;
}

inline std::string getUniqueFileID(Module &M) {
  llvm::sys::fs::UniqueID ID;
  if (auto EC = llvm::sys::fs::getUniqueID(M.getSourceFileName(), ID))
    PROTEUS_FATAL_ERROR("Could not get unique id for source file " +
                        EC.message());

  SmallString<64> Out;
  llvm::raw_svector_ostream OutStr(Out);
  OutStr << llvm::format("%x_%x", ID.getDevice(), ID.getFile());

  return std::string(Out);
}

inline bool isDeviceKernel(const Function *F) {
  if (!F)
    PROTEUS_FATAL_ERROR("Expected non-null function");

#if PROTEUS_ENABLE_CUDA
  const Module &M = *F->getParent();
  auto GetDeviceKernels = [&M]() {
    SmallPtrSet<Function *, 16> Kernels;
    NamedMDNode *MD = M.getNamedMetadata("nvvm.annotations");

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

    return Kernels;
  };

  // Create a kernel cache per module, assumes we don't insert/remove kernels
  // after parsing nvvm.annotations.
  static DenseMap<const Module *, SmallPtrSet<Function *, 16>> KernelCache;
  auto It = KernelCache.find(&M);
  if (It == KernelCache.end())
    It = KernelCache.insert({&M, GetDeviceKernels()}).first;
  const auto &KernelSet = It->second;
  if (KernelSet.contains(F))
    return true;

  return false;
#endif

#if PROTEUS_ENABLE_HIP
  return (F->getCallingConv() == CallingConv::AMDGPU_KERNEL);
#endif

  return false;
}

} // namespace proteus

namespace llvm {

using namespace proteus;

template <> struct DenseMapInfo<RuntimeConstantInfo> {
  static inline RuntimeConstantInfo getEmptyKey() {
    RuntimeConstantInfo K{BEGIN, -1};
    return K;
  }

  static inline RuntimeConstantInfo getTombstoneKey() {
    RuntimeConstantInfo K{END, -1};
    return K;
  }

  static unsigned getHashValue(const RuntimeConstantInfo &Val) {
    return hash_combine(Val.ArgInfo.Type, Val.ArgInfo.Pos);
  }

  static bool isEqual(const RuntimeConstantInfo &LHS,
                      const RuntimeConstantInfo &RHS) {
    return ((LHS.ArgInfo.Type == RHS.ArgInfo.Type) &&
            (LHS.ArgInfo.Pos == RHS.ArgInfo.Pos));
  }
};

} // namespace llvm

#endif
