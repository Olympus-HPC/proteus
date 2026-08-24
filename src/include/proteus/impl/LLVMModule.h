//===-- LLVMModule.h - Owned staged LLVM module API -------------*- C++ -*-===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_LLVM_MODULE_H
#define PROTEUS_LLVM_MODULE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace proteus {

class LLVMBackendUnavailableError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class LLVMSymbolNotFoundError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// A value snapshot of the Proteus options that affect staged LLVM
/// optimization, specialization policy, and device object generation.
struct LLVMCodeGenerationConfig {
  std::optional<std::string> Pipeline;
  std::string Method = "rtc";
  std::string OptLevel = "O3";
  unsigned CodegenOptLevel = 3;
  bool SpecializeArguments = true;
  bool SpecializeLaunchBounds = true;
  bool SpecializeDimensions = true;
  bool SpecializeDimensionRanges = true;
  std::optional<int> TunedMaxThreads;
  int MinBlocksPerSM = 0;
};

/// Return the effective process configuration for KernelName. An empty kernel
/// name selects the global configuration; a missing per-kernel entry also
/// falls back to the global configuration.
LLVMCodeGenerationConfig
getLLVMCodeGenerationConfig(const std::string &KernelName = "");

/// An owned mutable LLVM IR module. LLVM implementation types remain private
/// so callers can exchange modules without sharing LLVM handles or contexts.
class LLVMModule {
  class Impl;
  std::unique_ptr<Impl> PImpl;

  explicit LLVMModule(std::unique_ptr<Impl> PImpl);

public:
  using Dimensions = std::array<uint32_t, 3>;

  LLVMModule(LLVMModule &&) noexcept;
  LLVMModule &operator=(LLVMModule &&) noexcept;
  LLVMModule(const LLVMModule &) = delete;
  LLVMModule &operator=(const LLVMModule &) = delete;
  ~LLVMModule();

  static std::unique_ptr<LLVMModule> fromBitcode(const std::string &Bitcode);
  static std::unique_ptr<LLVMModule>
  fromIR(const std::string &IR, const std::string &Name = "<string>");
  static std::unique_ptr<LLVMModule>
  link(const std::vector<const LLVMModule *> &Modules);

  std::unique_ptr<LLVMModule> clone() const;
  std::string toBitcode() const;
  std::string toIR() const;
  void verify() const;

  LLVMModule &prune(bool UnsetExternallyInitialized = true);
  LLVMModule &internalize(const std::vector<std::string> &PreserveSymbols);

  LLVMModule &specializeArguments(const std::string &KernelName,
                                  void *const *Arguments,
                                  std::size_t NumArguments,
                                  const std::vector<std::size_t> &Indexes);
  LLVMModule &specializeLaunchDimensions(const Dimensions &Grid,
                                         const Dimensions &Block);
  LLVMModule &assumeLaunchDimensionRanges(const Dimensions &Grid,
                                          const Dimensions &Block);
  LLVMModule &setLaunchBounds(const std::string &KernelName,
                              unsigned MaxThreadsPerBlock,
                              unsigned MinBlocksPerSM);

  LLVMModule &optimize(const std::string &DeviceArch,
                       const LLVMCodeGenerationConfig &Config);
  std::string emitObject(const std::string &DeviceArch,
                         const LLVMCodeGenerationConfig &Config) const;
};

} // namespace proteus

#endif // PROTEUS_LLVM_MODULE_H
