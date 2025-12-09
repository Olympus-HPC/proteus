//===-- ObjectCacheChain.cpp -- Object cache chain implementation --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/Caching/ObjectCacheChain.h"

#ifdef PROTEUS_ENABLE_MPI
#include "proteus/Caching/MPISharedStorageCache.h"
#endif
#include "proteus/Caching/StorageCache.h"
#include "proteus/Config.h"
#include "proteus/Logger.h"
#include "proteus/TimeTracing.h"
#include "proteus/Utils.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>

#include <algorithm>
#include <cctype>

namespace proteus {

ObjectCacheChain::ObjectCacheChain(const std::string &Label)
    : Label(Label), DistributedRank(getDistributedRank()) {
  // Defer cache initialization to first use to allow MPI to be initialized.
}

void ObjectCacheChain::ensureInitialized() {
  if (Initialized)
    return;
  Initialized = true;
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
      if (Config::get().ProteusTraceOutput >= 1) {
        Logger::trace("[ObjectCacheChain] Added cache level: " + TrimmedName +
                      "\n");
      }
    }
  }

  if (Config::get().ProteusTraceOutput >= 1) {
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

#ifdef PROTEUS_ENABLE_MPI
  if (LowerName == "mpi-storage") {
    return std::make_unique<MPISharedStorageCache>(Label);
  }
#endif

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
  ensureInitialized();

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

      if (Config::get().ProteusTraceOutput >= 1) {
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
  ensureInitialized();

  for (auto &Cache : Caches) {
    Cache->store(HashValue, Entry);
  }
}

void ObjectCacheChain::printStats() {
  ensureInitialized();
  printf("[proteus][%s] ObjectCacheChain rank %s with %zu level(s):\n",
         Label.c_str(), DistributedRank.c_str(), Caches.size());
  for (auto &Cache : Caches) {
    Cache->printStats();
  }
}

void ObjectCacheChain::flush() {
  ensureInitialized();
  for (auto &Cache : Caches) {
    Cache->flush();
  }
}

} // namespace proteus
