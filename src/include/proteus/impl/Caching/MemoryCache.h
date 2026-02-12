//===-- MemoryCache.h -- In-memory code cache header implementation --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITCACHE_H
#define PROTEUS_JITCACHE_H

#include "proteus/impl/Config.h"
#include "proteus/impl/Hashing.h"
#include "proteus/impl/TimeTracing.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>

#include <cstdint>
#include <iostream>
#include <unordered_map>

namespace proteus {

using namespace llvm;

template <typename Function_t> class MemoryCache {
public:
  MemoryCache(const std::string &Label)
      : Label(Label), DistributedRank(getDistributedRank()) {}
  Function_t lookup(HashT &HashValue) {
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

  void insert(HashT &HashValue, Function_t FunctionPtr, StringRef FnName) {
    if (Config::get().ProteusDebugOutput) {
      if (CacheMap.count(HashValue))
        reportFatalError("MemoryCache collision detected");
    }

    auto &CacheEntry = CacheMap[HashValue];
    CacheEntry.FunctionPtr = FunctionPtr;
    CacheEntry.NumExecs = 1;
    CacheEntry.NumHits = 0;
    CacheEntry.FnName = FnName.str();
  }

  void printStats() {
    printf("[proteus][%s] MemoryCache rank %s hits %lu accesses %lu\n",
           Label.c_str(), DistributedRank.c_str(), Hits, Accesses);
    for (const auto &[HashValue, JCE] : CacheMap) {
      std::cout << "[proteus][" << Label << "] MemoryCache rank "
                << DistributedRank << " HashValue " << HashValue.toString()
                << " NumExecs " << JCE.NumExecs << " NumHits " << JCE.NumHits;
      if (Config::get().ProteusDebugOutput)
        std::cout << " FnName " << JCE.FnName;
      std::cout << "\n";
    }
  }

  void printKernelTrace() {
    if (!Config::get().traceKernels())
      return;

    if (CacheMap.empty())
      return;

    struct KernelStats {
      size_t Specializations = 0;
      uint64_t TotalLaunches = 0;
    };
    std::unordered_map<std::string, KernelStats> Grouped;
    for (const auto &[HashValue, JCE] : CacheMap) {
      auto &KS = Grouped[JCE.FnName];
      KS.Specializations++;
      KS.TotalLaunches += JCE.NumExecs;
    }

    printf("[proteus][%s] === Kernel Trace (rank %s) ===\n", Label.c_str(),
           DistributedRank.c_str());
    for (const auto &[MangledName, KS] : Grouped) {
      std::string Name = llvm::demangle(MangledName);
      printf("[proteus][%s]   %s  rank=%s  specializations=%zu  launches=%lu\n",
             Label.c_str(), Name.c_str(), DistributedRank.c_str(),
             KS.Specializations, KS.TotalLaunches);
    }
    printf("[proteus][%s] === End Kernel Trace ===\n", Label.c_str());
  }

private:
  struct MemoryCacheEntry {
    Function_t FunctionPtr;
    uint64_t NumExecs;
    uint64_t NumHits;
    std::string FnName;
  };

  std::unordered_map<HashT, MemoryCacheEntry> CacheMap;
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string Label;
  const std::string DistributedRank;
};

} // namespace proteus

#endif
