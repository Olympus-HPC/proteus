//===-- LLVMModule.cpp - Owned staged LLVM module API ---------------------===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/LLVMModule.h"

#include "proteus/impl/Config.h"
#include "proteus/impl/CoreLLVM.h"

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/impl/CoreLLVMDevice.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"
#include "proteus/impl/TransformArgumentSpecialization.h"
#endif

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/Internalize.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace proteus {
namespace {

using llvm::LLVMContext;
using llvm::MemoryBufferRef;
using llvm::Module;

std::unique_ptr<Module> parseBitcode(const std::string &Bitcode,
                                     LLVMContext &Context) {
  auto Parsed =
      llvm::parseBitcodeFile(MemoryBufferRef(Bitcode, "<bitcode>"), Context);
  if (!Parsed)
    throw std::runtime_error("failed to parse LLVM bitcode: " +
                             llvm::toString(Parsed.takeError()));
  return std::move(*Parsed);
}

std::unique_ptr<Module> parseIR(const std::string &IR, const std::string &Name,
                                LLVMContext &Context) {
  llvm::SMDiagnostic Diagnostic;
  auto Parsed = llvm::parseIR(MemoryBufferRef(IR, Name), Diagnostic, Context);
  if (!Parsed)
    throw std::runtime_error("failed to parse LLVM IR: " +
                             Diagnostic.getMessage().str());
  return Parsed;
}

std::string writeBitcode(const Module &M) {
  llvm::SmallString<0> Buffer;
  llvm::raw_svector_ostream Stream(Buffer);
  llvm::WriteBitcodeToFile(M, Stream);
  return std::string(Buffer.data(), Buffer.size());
}

std::string normalizeMethod(std::string Method) {
  std::transform(Method.begin(), Method.end(), Method.begin(),
                 [](unsigned char C) { return std::tolower(C); });
  if (Method != "rtc" && Method != "serial" && Method != "parallel")
    throw std::invalid_argument(
        "codegen method must be 'rtc', 'serial', or 'parallel'");
  return Method;
}

char parseOptLevel(const std::string &OptLevel) {
  if (OptLevel.size() != 2 || OptLevel[0] != 'O' ||
      std::string("0123sz").find(OptLevel[1]) == std::string::npos)
    throw std::invalid_argument(
        "optimization level must be O0, O1, O2, O3, Os, or Oz");
  return OptLevel[1];
}

void validateConfig(const LLVMCodeGenerationConfig &Config) {
  normalizeMethod(Config.Method);
  parseOptLevel(Config.OptLevel);
  if (Config.CodegenOptLevel > 3)
    throw std::invalid_argument(
        "codegen optimization level must be between 0 and 3");
  if (Config.TunedMaxThreads && *Config.TunedMaxThreads <= 0)
    throw std::invalid_argument("tuned max threads must be positive");
  if (Config.MinBlocksPerSM < 0)
    throw std::invalid_argument("minimum blocks per SM cannot be negative");
}

OptimizationPipelineConfig
makeOptimizationConfig(const LLVMCodeGenerationConfig &Config) {
  validateConfig(Config);
  return OptimizationPipelineConfig(
      Config.Pipeline, parseOptLevel(Config.OptLevel), Config.CodegenOptLevel);
}

CodegenOption parseCodegenOption(const std::string &Method) {
  const std::string Normalized = normalizeMethod(Method);
  if (Normalized == "rtc")
    return CodegenOption::RTC;
  if (Normalized == "serial")
    return CodegenOption::Serial;
  return CodegenOption::Parallel;
}

void validateDimensions(const LLVMModule::Dimensions &Grid,
                        const LLVMModule::Dimensions &Block) {
  const auto IsZero = [](uint32_t Value) { return Value == 0; };
  if (std::any_of(Grid.begin(), Grid.end(), IsZero) ||
      std::any_of(Block.begin(), Block.end(), IsZero))
    throw std::invalid_argument("launch dimensions must be positive");
}

void initializeLLVMTargets() {
  static InitLLVMTargets Initialize;
  (void)Initialize;
}

void validateOptimizationRequest(Module &M, const std::string &DeviceArch,
                                 const OptimizationPipelineConfig &Config) {
  auto TargetMachine =
      detail::createTargetMachine(M, DeviceArch, Config.CodegenOptLevel);
  if (!TargetMachine)
    throw std::runtime_error("failed to create LLVM target machine: " +
                             llvm::toString(TargetMachine.takeError()));

  if (!Config.PassPipeline)
    return;

  llvm::PassBuilder Passes(TargetMachine->get());
  std::unordered_set<std::string> LoadedPaths;
  for (const JITPassPluginConfig &PluginConfig : getJITPassPluginConfigs()) {
    if (!LoadedPaths.insert(PluginConfig.Path).second)
      continue;
    auto Plugin = llvm::PassPlugin::Load(PluginConfig.Path);
    if (!Plugin)
      throw std::runtime_error("failed to load LLVM pass plugin '" +
                               PluginConfig.Path +
                               "': " + llvm::toString(Plugin.takeError()));
    Plugin->registerPassBuilderCallbacks(Passes);
  }

  llvm::ModulePassManager ParsedPipeline;
  if (auto Error =
          Passes.parsePassPipeline(ParsedPipeline, *Config.PassPipeline))
    throw std::runtime_error("failed to parse LLVM pass pipeline: " +
                             llvm::toString(std::move(Error)));
}

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
RuntimeConstantType getRuntimeConstantType(llvm::Type *Type) {
  if (Type->isIntegerTy(1))
    return RuntimeConstantType::BOOL;
  if (Type->isIntegerTy(8))
    return RuntimeConstantType::INT8;
  if (Type->isIntegerTy(32))
    return RuntimeConstantType::INT32;
  if (Type->isIntegerTy(64))
    return RuntimeConstantType::INT64;
  if (Type->isFloatTy())
    return RuntimeConstantType::FLOAT;
  if (Type->isDoubleTy())
    return RuntimeConstantType::DOUBLE;
  if (Type->isFP128Ty() || Type->isPPC_FP128Ty() || Type->isX86_FP80Ty())
    return RuntimeConstantType::LONG_DOUBLE;
  if (Type->isPointerTy())
    return RuntimeConstantType::PTR;
  throw std::invalid_argument(
      "argument specialization supports scalar and pointer arguments only");
}

dim3 makeDim3(const LLVMModule::Dimensions &Dims) {
  dim3 Result;
  Result.x = Dims[0];
  Result.y = Dims[1];
  Result.z = Dims[2];
  return Result;
}
#endif

} // namespace

class LLVMModule::Impl {
public:
  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<Module> Mod;

  Impl(std::unique_ptr<LLVMContext> Context, std::unique_ptr<Module> Mod)
      : Context(std::move(Context)), Mod(std::move(Mod)) {}
};

LLVMCodeGenerationConfig
getLLVMCodeGenerationConfig(const std::string &KernelName) {
  const auto &Source = Config::get().getCGConfig(KernelName);
  LLVMCodeGenerationConfig Result;
  if (auto Pipeline = Source.optPipeline())
    Result.Pipeline = Pipeline.value();
  Result.Method = toString(Source.codeGenOption());
  std::transform(Result.Method.begin(), Result.Method.end(),
                 Result.Method.begin(),
                 [](unsigned char C) { return std::tolower(C); });
  Result.OptLevel = std::string("O") + Source.optLevel();
  Result.CodegenOptLevel = Source.codeGenOptLevel();
  Result.SpecializeArguments = Source.specializeArgs();
  Result.SpecializeLaunchBounds = Source.specializeLaunchBounds();
  Result.SpecializeDimensions = Source.specializeDims();
  Result.SpecializeDimensionRanges = Source.specializeDimsRange();
  if (Source.tunedMaxThreads() >= 0)
    Result.TunedMaxThreads = Source.tunedMaxThreads();
  Result.MinBlocksPerSM = Source.configuredMinBlocksPerSM();
  return Result;
}

LLVMModule::LLVMModule(std::unique_ptr<Impl> PImpl) : PImpl(std::move(PImpl)) {}
LLVMModule::LLVMModule(LLVMModule &&) noexcept = default;
LLVMModule &LLVMModule::operator=(LLVMModule &&) noexcept = default;
LLVMModule::~LLVMModule() = default;

std::unique_ptr<LLVMModule>
LLVMModule::fromBitcode(const std::string &Bitcode) {
  auto Context = std::make_unique<LLVMContext>();
  auto Module = parseBitcode(Bitcode, *Context);
  return std::unique_ptr<LLVMModule>(new LLVMModule(
      std::make_unique<Impl>(std::move(Context), std::move(Module))));
}

std::unique_ptr<LLVMModule> LLVMModule::fromIR(const std::string &IR,
                                               const std::string &Name) {
  auto Context = std::make_unique<LLVMContext>();
  auto Module = parseIR(IR, Name, *Context);
  return std::unique_ptr<LLVMModule>(new LLVMModule(
      std::make_unique<Impl>(std::move(Context), std::move(Module))));
}

std::unique_ptr<LLVMModule>
LLVMModule::link(const std::vector<const LLVMModule *> &Modules) {
  if (Modules.empty())
    throw std::invalid_argument("at least one LLVM module is required");
  if (std::any_of(Modules.begin(), Modules.end(),
                  [](const LLVMModule *M) { return M == nullptr; }))
    throw std::invalid_argument("cannot link a null LLVM module");

  auto Context = std::make_unique<LLVMContext>();
  auto Linked = std::make_unique<Module>("proteus.llvm.linked", *Context);
  llvm::Linker Linker(*Linked);
  for (const LLVMModule *Input : Modules) {
    auto Copy = parseBitcode(Input->toBitcode(), *Context);
    if (Linker.linkInModule(std::move(Copy)))
      throw std::runtime_error("failed to link LLVM module");
  }

  return std::unique_ptr<LLVMModule>(new LLVMModule(
      std::make_unique<Impl>(std::move(Context), std::move(Linked))));
}

std::unique_ptr<LLVMModule> LLVMModule::clone() const {
  return fromBitcode(toBitcode());
}

std::string LLVMModule::toBitcode() const { return writeBitcode(*PImpl->Mod); }

std::string LLVMModule::toIR() const {
  std::string IR;
  llvm::raw_string_ostream Stream(IR);
  PImpl->Mod->print(Stream, nullptr);
  return IR;
}

void LLVMModule::verify() const {
  std::string Diagnostic;
  llvm::raw_string_ostream Stream(Diagnostic);
  if (llvm::verifyModule(*PImpl->Mod, &Stream))
    throw std::runtime_error("LLVM module verification failed:\n" +
                             Stream.str());
}

LLVMModule &LLVMModule::prune(bool UnsetExternallyInitialized) {
  pruneIR(*PImpl->Mod, UnsetExternallyInitialized);
  return *this;
}

LLVMModule &
LLVMModule::internalize(const std::vector<std::string> &PreserveSymbols) {
  llvm::StringSet<> Preserve;
  for (const std::string &Name : PreserveSymbols) {
    if (!PImpl->Mod->getNamedValue(Name))
      throw LLVMSymbolNotFoundError("cannot preserve missing symbol '" + Name +
                                    "'");
    Preserve.insert(Name);
  }

  llvm::internalizeModule(*PImpl->Mod,
                          [&Preserve](const llvm::GlobalValue &GV) {
                            return Preserve.contains(GV.getName());
                          });
  return *this;
}

LLVMModule &LLVMModule::specializeArguments(
    const std::string &KernelName, void *const *Arguments,
    std::size_t NumArguments, const std::vector<std::size_t> &Indexes) {
#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  auto *Function = PImpl->Mod->getFunction(KernelName);
  if (!Function)
    throw LLVMSymbolNotFoundError("missing function '" + KernelName + "'");
  if (NumArguments != Function->arg_size())
    throw std::invalid_argument(
        "argument pointer-array length does not match function arity");
  if (!Arguments && NumArguments != 0)
    throw std::invalid_argument("argument pointer array cannot be null");

  std::unordered_set<std::size_t> Seen;
  std::vector<RuntimeConstantInfo> Infos;
  Infos.reserve(Indexes.size());
  for (std::size_t Index : Indexes) {
    if (Index >= NumArguments)
      throw std::invalid_argument(
          "argument specialization index is out of range");
    if (!Seen.insert(Index).second)
      throw std::invalid_argument(
          "argument specialization indexes must be unique");
    if (!Arguments[Index])
      throw std::invalid_argument(
          "selected argument storage pointer cannot be null");
    Infos.emplace_back(
        getRuntimeConstantType(Function->getArg(Index)->getType()), Index);
  }

  std::vector<RuntimeConstant> Constants;
  Constants.reserve(Infos.size());
  auto **MutableArguments = const_cast<void **>(Arguments);
  for (const RuntimeConstantInfo &Info : Infos)
    Constants.emplace_back(
        dispatchGetRuntimeConstantValue(MutableArguments, Info));

  TransformArgumentSpecialization::transform(*PImpl->Mod, *Function, Constants);
  return *this;
#else
  (void)KernelName;
  (void)Arguments;
  (void)NumArguments;
  (void)Indexes;
  throw LLVMBackendUnavailableError(
      "argument specialization requires a CUDA or HIP backend");
#endif
}

LLVMModule &LLVMModule::specializeLaunchDimensions(const Dimensions &Grid,
                                                   const Dimensions &Block) {
  validateDimensions(Grid, Block);
#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  dim3 GridDim = makeDim3(Grid);
  dim3 BlockDim = makeDim3(Block);
  setKernelDims(*PImpl->Mod, GridDim, BlockDim);
  return *this;
#else
  throw LLVMBackendUnavailableError(
      "launch-dimension specialization requires a CUDA or HIP backend");
#endif
}

LLVMModule &LLVMModule::assumeLaunchDimensionRanges(const Dimensions &Grid,
                                                    const Dimensions &Block) {
  validateDimensions(Grid, Block);
#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  dim3 GridDim = makeDim3(Grid);
  dim3 BlockDim = makeDim3(Block);
  setKernelDimsRange(*PImpl->Mod, GridDim, BlockDim);
  return *this;
#else
  throw LLVMBackendUnavailableError(
      "launch-dimension ranges require a CUDA or HIP backend");
#endif
}

LLVMModule &LLVMModule::setLaunchBounds(const std::string &KernelName,
                                        unsigned MaxThreadsPerBlock,
                                        unsigned MinBlocksPerSM) {
  if (MaxThreadsPerBlock == 0 || MaxThreadsPerBlock > 1024)
    throw std::invalid_argument(
        "maximum threads per block must be between 1 and 1024");
#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  auto *Function = PImpl->Mod->getFunction(KernelName);
  if (!Function)
    throw LLVMSymbolNotFoundError("missing function '" + KernelName + "'");
#if defined(PROTEUS_ENABLE_CUDA) && LLVM_VERSION_MAJOR < 22
  if (!PImpl->Mod->getNamedMetadata("nvvm.annotations"))
    throw std::invalid_argument(
        "CUDA launch bounds require nvvm.annotations metadata");
#endif
  setLaunchBoundsForKernel(*Function, MaxThreadsPerBlock, MinBlocksPerSM);
  return *this;
#else
  (void)KernelName;
  (void)MinBlocksPerSM;
  throw LLVMBackendUnavailableError(
      "launch bounds require a CUDA or HIP backend");
#endif
}

LLVMModule &LLVMModule::optimize(const std::string &DeviceArch,
                                 const LLVMCodeGenerationConfig &Config) {
  if (DeviceArch.empty())
    throw std::invalid_argument("device architecture cannot be empty");
  initializeLLVMTargets();
  const auto Optimization = makeOptimizationConfig(Config);
  validateOptimizationRequest(*PImpl->Mod, DeviceArch, Optimization);
  optimizeIR(*PImpl->Mod, DeviceArch, Optimization);
  return *this;
}

std::string
LLVMModule::emitObject(const std::string &DeviceArch,
                       const LLVMCodeGenerationConfig &Config) const {
  if (DeviceArch.empty())
    throw std::invalid_argument("device architecture cannot be empty");
  validateConfig(Config);
  initializeLLVMTargets();
  auto Copy = clone();
  const auto Optimization = makeOptimizationConfig(Config);
  validateOptimizationRequest(*Copy->PImpl->Mod, DeviceArch, Optimization);

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  const CodegenOption Method = parseCodegenOption(Config.Method);
#if defined(PROTEUS_ENABLE_CUDA)
  if (Method != CodegenOption::RTC)
    throw std::invalid_argument(
        "CUDA object emission supports only the 'rtc' method");
  auto Object = codegenObject(*Copy->PImpl->Mod, DeviceArch,
                              GlobalLinkedBinaries, Method, Optimization);
#else
  auto Object = codegenObject(*Copy->PImpl->Mod, DeviceArch,
                              GlobalLinkedBinaries, Method, Optimization);
#endif
  if (!Object)
    throw std::runtime_error("device object emission returned no data");
  llvm::StringRef Buffer = Object->getBuffer();
  return std::string(Buffer.data(), Buffer.size());
#else
  (void)Copy;
  throw LLVMBackendUnavailableError(
      "device object emission requires a CUDA or HIP backend");
#endif
}

} // namespace proteus
