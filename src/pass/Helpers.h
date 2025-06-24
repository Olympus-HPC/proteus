#ifndef PROTEUS_PASS_HELPERS_H
#define PROTEUS_PASS_HELPERS_H

#include <llvm/ADT/SetVector.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>

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
  SmallSetVector<int, 16> ConstantArgs;
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

} // namespace proteus

#endif
