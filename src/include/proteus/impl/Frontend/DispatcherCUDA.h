#ifndef PROTEUS_FRONTEND_DISPATCHER_CUDA_H
#define PROTEUS_FRONTEND_DISPATCHER_CUDA_H

#if PROTEUS_ENABLE_CUDA

#include "proteus/Error.h"
#include "proteus/impl/Frontend/CUDAToolchain.h"
#include "proteus/impl/Frontend/DispatcherDevice.h"
#include "proteus/impl/JitEngineDeviceCUDA.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

class DispatcherCUDA : public DispatcherDevice<JitEngineDeviceCUDA> {
public:
  static DispatcherCUDA &instance() {
    static DispatcherCUDA D{"DispatcherCUDA", JitEngineDeviceCUDA::instance()};
    return D;
  }

  DispatcherCUDA(const std::string &Label, JitEngineDeviceCUDA &Jit)
      : DispatcherDevice(Label, TargetModelType::CUDA, Jit) {}

protected:
  void linkDeviceLibraries(Module &M) override {
    TIMESCOPE(DispatcherCUDA, linkDeviceLibraries);
    const auto &Toolchain = resolveCUDAToolchain();
    auto LibDeviceBuffer = llvm::MemoryBuffer::getFile(Toolchain.LibDevicePath);
    if (!LibDeviceBuffer || !LibDeviceBuffer.get())
      reportFatalError("DispatchCUDA: failed to read libdevice from " +
                       Toolchain.LibDevicePath + " (" + Toolchain.Origin + ")");

    auto LibDeviceModule = llvm::parseBitcodeFile(
        LibDeviceBuffer->get()->getMemBufferRef(), M.getContext());
    if (!LibDeviceModule)
      reportFatalError("DispatchCUDA: failed to parse libdevice from " +
                       Toolchain.LibDevicePath + " (" + Toolchain.Origin + ")");

    llvm::Linker Linker(M);
    Linker.linkInModule(std::move(LibDeviceModule.get()),
                        llvm::Linker::Flags::LinkOnlyNeeded);
  }
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_CUDA_H
