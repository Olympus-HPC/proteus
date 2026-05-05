//===-- CompilerInterfaceDevice.cpp -- JIT library entry point for GPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/CompilerInterfaceDevice.h"
#include "proteus/RecordInterface.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/CompilerInterfaceDeviceInternal.h"
#include "proteus/impl/JitEngineDevice.h"
#include "proteus/impl/JitEngineInfoRegistry.h"

#include <memory>
#include <string>
#include <vector>

using namespace proteus;

struct ProteusRecordedKernel {
  struct Global {
    std::string Name;
    const void *HostAddr;
    const void *DevAddr;
    uint64_t Size;
  };

  std::string Name;
  uint64_t StaticHash;
  std::string Bitcode;
  std::vector<Global> Globals;
};

// NOLINTBEGIN(readability-identifier-naming)

// NOTE: A great mystery is: why does this work ONLY if HostAddr is a CONST
// void* for HIP
extern "C" __attribute((used)) void __proteus_register_var(void *Handle,
                                                           const void *HostAddr,
                                                           const char *VarName,
                                                           uint64_t VarSize) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  // NOTE: For HIP it works to get the symbol address during the call inside a
  // constructor context, but for CUDA, it fails.  So we save the host address
  // and defer resolving the symbol address when patching the bitcode, which
  // works for both CUDA and HIP.
  JitEngineInfo.registerVar(Handle, HostAddr, VarName, VarSize);
}

extern "C" __attribute__((used)) void
__proteus_register_fatbinary(void *Handle, void *FatbinWrapper,
                             const char *ModuleId) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFatBinary(Handle, FatbinWrapper, ModuleId);
}

extern "C" __attribute__((used)) void
__proteus_register_fatbinary_end(void *Handle) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFatBinaryEnd(Handle);
}

extern "C" __attribute__((used)) void
__proteus_register_linked_binary(void *FatbinWrapper, const char *ModuleId) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerLinkedBinary(FatbinWrapper, ModuleId);
}

extern "C" __attribute((used)) void
__proteus_register_function(void *Handle, void *Kernel, char *KernelName,
                            RuntimeConstantInfo **RCInfoArrayPtr,
                            int32_t NumRCs) {
  ArrayRef<RuntimeConstantInfo *> RCInfoArray{RCInfoArrayPtr,
                                              static_cast<size_t>(NumRCs)};
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFunction(Handle, Kernel, KernelName, RCInfoArray);
}

extern "C" proteus::DeviceTraits<JitDeviceImplT>::DeviceError_t
__proteus_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                        void **KernelArgs, uint64_t ShmemSize, void *Stream) {
  TIMESCOPE("__proteus_launch_kernel");
  return __proteus_launch_kernel_internal(Kernel, GridDim, BlockDim, KernelArgs,
                                          ShmemSize, Stream);
}

extern "C" void __proteus_enable_device() {
  auto &Jit = JitDeviceImplT::instance();
  Jit.enable();
}

extern "C" void __proteus_disable_device() {
  auto &Jit = JitDeviceImplT::instance();
  Jit.disable();
}

extern "C" ProteusRecordStatus
__proteus_record_capture_kernel(const void *Kernel,
                                ProteusRecordedKernel **Out) {
  if (!Out)
    return PROTEUS_RECORD_ERROR;

  *Out = nullptr;
  auto &Jit = JitDeviceImplT::instance();
  auto OptionalKernelInfo = Jit.getJITKernelInfo(Kernel);
  if (!OptionalKernelInfo)
    return PROTEUS_RECORD_KERNEL_NOT_FOUND;

  auto &KInfo = OptionalKernelInfo.value().get();
  Jit.extractModuleAndBitcode(KInfo);
  auto StaticHash = Jit.getStaticHash(KInfo);
  auto Bitcode = KInfo.getBitcode().getBuffer();
  auto &GlobalVars = KInfo.getBinaryInfo().getVarNameToGlobalInfo();

  auto Record = std::make_unique<ProteusRecordedKernel>();
  Record->Name = KInfo.getName();
  Record->StaticHash = StaticHash.getValue();
  Record->Bitcode.assign(Bitcode.data(), Bitcode.size());
  Record->Globals.reserve(GlobalVars.size());
  for (const auto &[Name, GV] : GlobalVars)
    Record->Globals.push_back({Name, GV.HostAddr, GV.DevAddr, GV.VarSize});

  *Out = Record.release();
  return PROTEUS_RECORD_OK;
}

extern "C" void __proteus_record_release_kernel(ProteusRecordedKernel *Record) {
  delete Record;
}

extern "C" const char *
__proteus_record_kernel_name(const ProteusRecordedKernel *Record) {
  if (!Record)
    return nullptr;
  return Record->Name.c_str();
}

extern "C" uint64_t
__proteus_record_static_hash(const ProteusRecordedKernel *Record) {
  if (!Record)
    return 0;
  return Record->StaticHash;
}

extern "C" const void *
__proteus_record_bitcode_data(const ProteusRecordedKernel *Record) {
  if (!Record)
    return nullptr;
  return Record->Bitcode.data();
}

extern "C" size_t
__proteus_record_bitcode_size(const ProteusRecordedKernel *Record) {
  if (!Record)
    return 0;
  return Record->Bitcode.size();
}

extern "C" size_t
__proteus_record_global_count(const ProteusRecordedKernel *Record) {
  if (!Record)
    return 0;
  return Record->Globals.size();
}

extern "C" ProteusRecordedGlobal
__proteus_record_global_at(const ProteusRecordedKernel *Record, size_t Index) {
  if (!Record || Index >= Record->Globals.size())
    return {nullptr, nullptr, nullptr, 0};

  const auto &Global = Record->Globals[Index];
  return {Global.Name.c_str(), Global.HostAddr, Global.DevAddr, Global.Size};
}

// NOLINTEND(readability-identifier-naming)
