// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include "proteus/Error.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>

#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>

#include "proteus/CoreLLVMHIP.hpp"
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/InitLLVM.h>
#include <string>

// TODO: To run execute <exe> _Z3fooi

int main(int argc, char **argv) {
  proteus::InitNativeTarget();
  std::string DeviceArch;
  hipDeviceProp_t DevProp;
  proteus::InitAMDGPUTarget();
  proteusHipErrCheck(hipGetDeviceProperties(&DevProp, 0));

  DeviceArch = DevProp.gcnArchName;
  DeviceArch = DeviceArch.substr(0, DeviceArch.find_first_of(":"));

  std::string BitcodeFN(argv[1]);
  llvm::LLVMContext Ctx;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(BitcodeFN);
  if (!Buffer)
    PROTEUS_FATAL_ERROR("Error with loading file " + BitcodeFN +
                        "\n Error Code:" + Buffer.getError().message());

  llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
      llvm::parseBitcodeFile(Buffer->get()->getMemBufferRef(), Ctx);

  if (!ModuleOrErr)
    PROTEUS_FATAL_ERROR("Error parsing bitcode: " +
                        llvm::toString(ModuleOrErr.takeError()));

  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto Mod = std::move(ModuleOrErr.get());
  proteus::pruneIR(*Mod);
  auto KernelFunc = Mod->getFunction(argv[2]);
  proteus::internalize(*Mod, KernelFunc->getName());
  proteus::optimizeIR(*Mod, DeviceArch, '1', 1);
  auto DeviceObject =
      proteus::codegenObject(*Mod, DeviceArch, GlobalLinkedBinaries);
  if (!DeviceObject)
    return -1;

  std::cout << "Success \n";
  return 0;
}
