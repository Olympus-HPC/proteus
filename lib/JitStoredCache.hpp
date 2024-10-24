//===-- JitStoredCache.hpp -- JIT storage-based cache header impl. --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITSTOREDCACHE_HPP
#define PROTEUS_JITSTOREDCACHE_HPP

#include <cstdint>
#include <filesystem>

#include "llvm/ADT/StringRef.h"

#include "Utils.h"

namespace proteus {

using namespace llvm;

// NOTE: Stored cache assumes that stored code is re-usable across runs!
// Source code changes should invalidate the cache (TODO). Also, if
// storing assembly (PTX) or binary (ELF), then device globals may
// have different addresses that render it invalid. In this case, store LLVM IR
// to re-link globals.
template <typename Function_t> class JitStoredCache {
public:
  JitStoredCache() { std::filesystem::create_directory(StorageDirectory); }
  template <typename TryLoadLambdaT>
  Function_t lookup(uint64_t HashValue, StringRef Kernel,
                    TryLoadLambdaT &&TryLoadLambda) {
    TIMESCOPE("object lookup");
    Accesses++;

    Function_t DevFunction;
#if ENABLE_LLVMIR_STORED_CACHE
#error Unsupported yet
#endif
    DevFunction = TryLoadLambda(
        (StorageDirectory + "/cache-jit-" + std::to_string(HashValue) + ".o"),
        Kernel);

    if (!DevFunction)
      return nullptr;

    Hits++;
    return DevFunction;
  }

  void storeObject(uint64_t HashValue, StringRef ObjectRef) {
    TIMESCOPE("Store object");
    std::error_code EC;
    raw_fd_ostream OutBin(Twine(StorageDirectory + "/cache-jit-" +
                                std::to_string(HashValue) + ".o")
                              .str(),
                          EC);
    if (EC)
      FATAL_ERROR("Cannot open device object file");
    OutBin << ObjectRef;
    OutBin.close();
  }

  void printStats() {
    // Use printf to avoid re-ordering outputs by outs() in HIP.
    printf("JitStoredCache hits %lu total %lu\n", Hits, Accesses);
  }

private:
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory = ".proteus";
};

} // namespace proteus

#endif