#ifndef PROTEUS_FRONTEND_DISPATCHER_HIP_H
#define PROTEUS_FRONTEND_DISPATCHER_HIP_H

#if PROTEUS_ENABLE_HIP

#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/impl/Caching/ObjectCacheChain.h"
#include "proteus/impl/JitEngineDeviceHIP.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

class DispatcherHIP : public Dispatcher {
public:
  static DispatcherHIP &instance() {
    static DispatcherHIP D;
    return D;
  }

  std::unique_ptr<MemoryBuffer> compile(std::unique_ptr<LLVMContext> Ctx,
                                        std::unique_ptr<Module> Mod,
                                        const HashT &ModuleHash,
                                        bool DisableIROpt = false) override {
    // This is necessary to ensure Ctx outlives M. Setting [[maybe_unused]] can
    // trigger a lifetime bug.
    auto CtxOwner = std::move(Ctx);
    auto ModOwner = std::move(Mod);

    auto LoadBitcode = [&](const llvm::SmallString<256> &Path) {
      auto BufferOrErr = llvm::MemoryBuffer::getFile(Path);
      if (!BufferOrErr || !BufferOrErr.get())
        reportFatalError("DispatchHIP: failed to read ROCm bitcode file: " +
                         Path.str().str());
      auto Parsed = llvm::parseBitcodeFile(
          BufferOrErr->get()->getMemBufferRef(), ModOwner->getContext());
      if (!Parsed)
        reportFatalError("DispatchHIP: failed to parse ROCm bitcode file: " +
                         Path.str().str());
      return std::move(Parsed.get());
    };

    auto AppendBitcodePath =
        [&](llvm::SmallVectorImpl<llvm::SmallString<256>> &Paths,
            llvm::StringRef Filename) {
          llvm::SmallString<256> Path{PROTEUS_ROCM_BITCODE_DIR};
          llvm::sys::path::append(Path, Filename);
          Paths.push_back(std::move(Path));
        };

    auto Exists = [&](llvm::StringRef Filename) -> bool {
      llvm::SmallString<256> Path{PROTEUS_ROCM_BITCODE_DIR};
      llvm::sys::path::append(Path, Filename);
      return llvm::sys::fs::exists(Path);
    };

    auto PickFirstExisting =
        [&](std::initializer_list<llvm::StringRef> Candidates)
        -> llvm::StringRef {
      for (auto C : Candidates) {
        if (Exists(C))
          return C;
      }
      return {};
    };

    // Link ROCm device libraries (ocml/ockl + oclc config) so HIPRTC can
    // resolve __ocml_* calls produced by math lowering.
    llvm::SmallVector<llvm::SmallString<256>, 8> LibsToLink;
    AppendBitcodePath(LibsToLink, "ocml.bc");
    AppendBitcodePath(LibsToLink, "ockl.bc");

    // ABI: prefer the newest available.
    if (auto Abi = PickFirstExisting({"oclc_abi_version_600.bc",
                                      "oclc_abi_version_500.bc",
                                      "oclc_abi_version_400.bc"});
        !Abi.empty()) {
      AppendBitcodePath(LibsToLink, Abi);
    } else {
      reportFatalError(
          std::string("DispatchHIP: missing oclc ABI bitcode under ") +
          PROTEUS_ROCM_BITCODE_DIR +
          " (expected oclc_abi_version_{600,500,400}.bc)");
    }

    // ISA: derived from device arch like "gfx90a" -> "90a".
    const std::string DeviceArch = Jit.getDeviceArch().str();
    if (!llvm::StringRef{DeviceArch}.starts_with("gfx"))
      reportFatalError("DispatchHIP: unexpected HIP device arch: " +
                       DeviceArch);
    const llvm::StringRef IsaSuffix = llvm::StringRef{DeviceArch}.drop_front(3);
    const std::string IsaFile = ("oclc_isa_version_" + IsaSuffix + ".bc").str();
    if (!Exists(IsaFile))
      reportFatalError(std::string("DispatchHIP: missing ISA bitcode file ") +
                       IsaFile + " under " + PROTEUS_ROCM_BITCODE_DIR +
                       " (DeviceArch=" + DeviceArch + ")");
    AppendBitcodePath(LibsToLink, IsaFile);

    // Math/FP mode defaults (safe defaults, can be revisited later).
    AppendBitcodePath(LibsToLink, "oclc_unsafe_math_off.bc");
    AppendBitcodePath(LibsToLink, "oclc_finite_only_off.bc");
    AppendBitcodePath(LibsToLink, "oclc_daz_opt_off.bc");
    AppendBitcodePath(LibsToLink, "oclc_correctly_rounded_sqrt_on.bc");

    // Wavefront size selection: RDNA is typically wave32; CDNA/gfx9 wave64.
    const bool IsWave32 = llvm::StringRef{DeviceArch}.starts_with("gfx10") ||
                          llvm::StringRef{DeviceArch}.starts_with("gfx11") ||
                          llvm::StringRef{DeviceArch}.starts_with("gfx12");
    AppendBitcodePath(LibsToLink, IsWave32 ? "oclc_wavefrontsize64_off.bc"
                                           : "oclc_wavefrontsize64_on.bc");

    llvm::Linker Linker{*ModOwner};
    for (const auto &Path : LibsToLink) {
      auto LibMod = LoadBitcode(Path);
      Linker.linkInModule(std::move(LibMod),
                          llvm::Linker::Flags::LinkOnlyNeeded);
    }

    std::unique_ptr<MemoryBuffer> ObjectModule =
        Jit.compileOnly(*ModOwner, DisableIROpt);
    if (!ObjectModule)
      reportFatalError("Expected non-null object library");

    ObjectCache->store(
        ModuleHash, CacheEntry::staticObject(ObjectModule->getMemBufferRef()));

    return ObjectModule;
  }

  std::unique_ptr<CompiledLibrary>
  lookupCompiledLibrary(const HashT &ModuleHash) override {
    return ObjectCache->lookup(ModuleHash);
  }

  DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                        LaunchDims BlockDim, void *KernelArgs[],
                        uint64_t ShmemSize, void *Stream) override {
    dim3 HipGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 HipBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    hipStream_t HipStream = reinterpret_cast<hipStream_t>(Stream);

    return proteus::launchKernelFunction(
        reinterpret_cast<hipFunction_t>(KernelFunc), HipGridDim, HipBlockDim,
        KernelArgs, ShmemSize, HipStream);
  }

  StringRef getDeviceArch() const override { return Jit.getDeviceArch(); }

  ~DispatcherHIP() {
    CodeCache.printStats();
    CodeCache.printKernelTrace();
    ObjectCache->printStats();
  }

  void *getFunctionAddress(const std::string &KernelName,
                           const HashT &ModuleHash,
                           CompiledLibrary &Library) override {
    auto GetKernelFunc = [&]() {
      // Hash the kernel name to get a unique id.
      HashT HashValue = hash(KernelName, ModuleHash);

      if (auto KernelFunc = CodeCache.lookup(HashValue))
        return KernelFunc;

      auto KernelFunc = proteus::getKernelFunctionFromImage(
          KernelName, Library.ObjectModule->getBufferStart(),
          /*RelinkGlobalsByCopy*/ false,
          /* VarNameToGlobalInfo */ {});

      CodeCache.insert(HashValue, KernelFunc, KernelName);

      return KernelFunc;
    };

    auto KernelFunc = GetKernelFunc();
    return KernelFunc;
  }

  void registerDynamicLibrary(const HashT &, const std::string &) override {
    reportFatalError("Dispatch HIP does not support registerDynamicLibrary");
  }

  void registerObject(const HashT &HashValue,
                      const llvm::MemoryBufferRef &Obj) override {
    ObjectCache->store(HashValue, CacheEntry::staticObject(Obj));
  }

private:
  JitEngineDeviceHIP &Jit;
  DispatcherHIP()
      : Dispatcher("DispatcherHIP", TargetModelType::HIP),
        Jit(JitEngineDeviceHIP::instance()) {}
  MemoryCache<hipFunction_t> CodeCache{"DispatcherHIP"};
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_HIP_H
