//===-- ObjectCacheChain.cpp -- Object cache chain implementation --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/Caching/ObjectCacheChain.h"
#ifdef PROTEUS_ENABLE_MPI
#include "proteus/impl/Caching/MPILocalLookupCache.h"
#include "proteus/impl/Caching/MPIRemoteLookupCache.h"
#endif
#include "proteus/impl/Caching/StorageCache.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/TimeTracing.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>

#include <algorithm>
#include <cctype>

namespace proteus {

ObjectCacheChain::ObjectCacheChain(const std::string &Label)
    : Label(Label), DistributedRank(getDistributedRank()) {
  buildFromConfig(Config::get().ProteusObjectCacheChain);
}

void ObjectCacheChain::buildFromConfig(const std::string &ConfigStr) {
  // Parse comma-separated list of cache names.
  StringRef ConfigRef(ConfigStr);
  SmallVector<StringRef, 4> CacheNames;
  ConfigRef.split(CacheNames, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  for (const auto &Name : CacheNames) {
    std::string TrimmedName = Name.trim().str();
    if (auto Cache = createCache(TrimmedName)) {
      addCache(std::move(Cache));
      if (Config::get().traceSpecializations()) {
        Logger::trace("[ObjectCacheChain] Added cache level: " + TrimmedName +
                      "\n");
      }
    }
  }

  if (Config::get().traceSpecializations()) {
    std::string ChainDesc = "[ObjectCacheChain] Chain for " + Label + ": ";
    for (size_t I = 0; I < Caches.size(); ++I) {
      if (I > 0)
        ChainDesc += " -> ";
      ChainDesc += Caches[I]->getName();
    }
    ChainDesc += "\n";
    Logger::trace(ChainDesc);
  }
}

std::unique_ptr<ObjectCache>
ObjectCacheChain::createCache(const std::string &Name) {
  // Normalize name to lowercase.
  std::string LowerName = Name;
  std::transform(LowerName.begin(), LowerName.end(), LowerName.begin(),
                 [](unsigned char C) { return std::tolower(C); });

  if (LowerName == "storage") {
    return std::make_unique<StorageCache>(Label);
  }

  if (LowerName == "mpi-local-lookup") {
#ifdef PROTEUS_ENABLE_MPI
    return std::make_unique<MPILocalLookupCache>(Label);
#else
    reportFatalError(
        "MPILocalLookupCache requested but Proteus built without MPI");
#endif
  }

  if (LowerName == "mpi-remote-lookup") {
#ifdef PROTEUS_ENABLE_MPI
    return std::make_unique<MPIRemoteLookupCache>(Label);
#else
    reportFatalError(
        "MPIRemoteLookupCache requested but Proteus built without MPI");
#endif
  }

  reportFatalError("Unknown cache type: " + Name);
  return nullptr;
}

void ObjectCacheChain::addCache(std::unique_ptr<ObjectCache> Cache) {
  if (Cache) {
    Caches.push_back(std::move(Cache));
  }
}

void ObjectCacheChain::promoteToLevel(const HashT &HashValue,
                                      const CacheEntry &Entry, size_t Level) {
  for (size_t J = 0; J < Level; ++J) {
    Caches[J]->store(HashValue, Entry);
  }
}

std::unique_ptr<CompiledLibrary>
ObjectCacheChain::lookup(const HashT &HashValue) {
  TIMESCOPE("ObjectCacheChain::lookup");

  // Search from fastest (index 0) to slowest.
  for (size_t I = 0; I < Caches.size(); ++I) {
    if (auto Result = Caches[I]->lookup(HashValue)) {
      // Populate higher-level caches with the found entry.
      if (I > 0) {
        if (Result->isStaticObject()) {
          promoteToLevel(
              HashValue,
              CacheEntry::staticObject(Result->ObjectModule->getMemBufferRef()),
              I);
        } else if (Result->isSharedObject()) {
          // Read the .so file and promote to higher-level caches.
          auto Buf = MemoryBuffer::getFileAsStream(Result->DynLibPath);
          if (Buf) {
            promoteToLevel(HashValue,
                           CacheEntry::sharedObject((*Buf)->getMemBufferRef()),
                           I);
          }
        }
      }

      if (Config::get().traceSpecializations()) {
        Logger::trace("[ObjectCacheChain] Hit at level " + std::to_string(I) +
                      " (" + Caches[I]->getName() + ") for hash " +
                      HashValue.toString() + "\n");
      }

      return Result;
    }
  }

  return nullptr;
}

void ObjectCacheChain::store(const HashT &HashValue, const CacheEntry &Entry) {
  TIMESCOPE("ObjectCacheChain::store");

  for (auto &Cache : Caches) {
    Cache->store(HashValue, Entry);
  }
}

void ObjectCacheChain::printStats() {
  printf("[proteus][%s] ObjectCacheChain rank %s with %zu level(s):\n",
         Label.c_str(), DistributedRank.c_str(), Caches.size());
  for (auto &Cache : Caches) {
    Cache->printStats();
  }
}

void ObjectCacheChain::finalize() {
  for (auto &Cache : Caches) {
    Cache->finalize();
  }
}

} // namespace proteus
