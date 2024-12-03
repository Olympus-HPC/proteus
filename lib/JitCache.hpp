//===-- JitCache.hpp -- JIT in-memory code cache header implementation --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITCACHE_HPP
#define PROTEUS_JITCACHE_HPP

#include <cstdint>
#include <filesystem>
#include <iostream>

#include "CompilerInterfaceTypes.h"
#include "Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

inline llvm::hash_code hash_value(const RuntimeConstant &RC) {
  return llvm::hash_value(RC.Value.Int64Val);
}

namespace proteus {

using namespace llvm;

template <typename Function_t> class JitCache {
public:
  uint64_t hash(StringRef ModuleUniqueId, StringRef FnName,
                const RuntimeConstant *RC, int NumRuntimeConstants) const {
    ArrayRef<RuntimeConstant> Data(RC, NumRuntimeConstants);
    auto HashValue = hash_combine(ExePath, ModuleUniqueId, FnName, Data);
    return HashValue;
  }

  Function_t lookup(uint64_t HashValue) {
    TIMESCOPE("lookup");
    Accesses++;

    auto It = CacheMap.find(HashValue);
    if (It == CacheMap.end())
      return nullptr;

    It->second.NumExecs++;
    It->second.NumHits++;
    Hits++;
    return It->second.FunctionPtr;
  }

  void insert(uint64_t HashValue, Function_t FunctionPtr, StringRef FnName,
              RuntimeConstant *RC, int NumRuntimeConstants) {
#if ENABLE_DEBUG
    if (CacheMap.count(HashValue))
      FATAL_ERROR("JitCache collision detected");
#endif

    CacheMap[HashValue] = {FunctionPtr, /* num_execs */ 1};

#if ENABLE_DEBUG
    CacheMap[HashValue].FnName = FnName.str();
    for (size_t I = 0; I < NumRuntimeConstants; ++I)
      CacheMap[HashValue].RCVector.push_back(RC[I]);
#endif
  }

  void printStats() {
    // outs() << "JitCache hits " << Hits << " total " << Accesses << "\n";
    // Use printf to avoid re-ordering outputs by outs() in HIP.
    printf("JitCache hits %lu total %lu\n", Hits, Accesses);
    for (auto &It : CacheMap) {
      uint64_t HashValue = It.first;
      JitCacheEntry &JCE = It.second;
      // outs() << "HashValue " << HashValue << " num_execs " <<
      // JCE.NumExecs;
      printf("HashValue %lu NumExecs %lu NumHits %lu", HashValue, JCE.NumExecs,
             JCE.NumHits);
#if ENABLE_DEBUG
      // outs() << " FnName " << JCE.FnName << " RCs [";
      printf(" FnName %s RCs [", JCE.FnName.c_str());
      for (auto &RC : JCE.RCVector)
        // outs() << RC.Int64Val << ", ";
        printf("%ld, ", RC.Value.Int64Val);
      // outs() << "]";
      printf("]");
#endif
      // outs() << "\n";
      printf("\n");
    }
  }

  JitCache() {
    // NOTE: Linux-specific.
    ExePath = std::filesystem::canonical("/proc/self/exe");
  }

private:
  struct JitCacheEntry {
    Function_t FunctionPtr;
    uint64_t NumExecs;
    uint64_t NumHits;
#if ENABLE_DEBUG
    std::string FnName;
    SmallVector<RuntimeConstant, 8> RCVector;
#endif
  };

  DenseMap<uint64_t, JitCacheEntry> CacheMap;
  // Use the executable binary path when hashing to differentiate between
  // same-named kernels generated by other executables.
  std::filesystem::path ExePath;
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
};

} // namespace proteus

#endif