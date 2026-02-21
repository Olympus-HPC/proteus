#include <cstdlib>
#include <iostream>
#include <string>

#include <proteus/Error.h>
#include <proteus/JitInterface.h>
#include <proteus/impl/CoreLLVM.h>
#include <proteus/impl/CoreLLVMDevice.h>
#include <proteus/impl/JitEngineDeviceCUDA.h>
#include <proteus/impl/Utils.h>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/InitLLVM.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./main <bitcode> <kernel symbol>\n";
    return 1;
  }

  std::string BitcodeFN(argv[1]);
  const char *KernelSym = argv[2];

  proteus::InitLLVMTargets Init;

  proteus::JitEngineDeviceCUDA &Jit = proteus::JitEngineDeviceCUDA::instance();

  llvm::StringRef DeviceArch = Jit.getDeviceArch();

  llvm::LLVMContext Ctx;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(BitcodeFN);
  if (!Buffer)
    proteus::reportFatalError("Error loading file " + BitcodeFN +
                              "\n Error Code:" + Buffer.getError().message());

  llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
      llvm::parseBitcodeFile(Buffer->get()->getMemBufferRef(), Ctx);

  if (!ModuleOrErr)
    proteus::reportFatalError("Error parsing bitcode: " +
                              llvm::toString(ModuleOrErr.takeError()));

  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto Mod = std::move(ModuleOrErr.get());
  proteus::pruneIR(*Mod);
  proteus::internalize(*Mod, KernelSym);
  proteus::optimizeIR(*Mod, DeviceArch, '1', 1);
  auto DeviceObject =
      proteus::codegenObject(*Mod, DeviceArch, GlobalLinkedBinaries);
  if (!DeviceObject)
    return -1;

  auto JitKernelFunc = proteus::getKernelFunctionFromImage(
      KernelSym, DeviceObject->getBufferStart(), false, {});

  int Arg = 42;
  void *KernelArgs[] = {&Arg};
  proteusCudaErrCheck(proteus::launchKernelFunction(
      JitKernelFunc, dim3{1}, dim3{1}, KernelArgs, 0, nullptr));

  proteusCudaErrCheck(cudaDeviceSynchronize());

  std::cout << "Success \n";

  return 0;
}
