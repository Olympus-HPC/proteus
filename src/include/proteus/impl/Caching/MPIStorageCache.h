//===-- MPIStorageCache.h -- MPI storage cache base class header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for MPI-based storage caches. Provides shared infrastructure for
// store forwarding, disk persistence, pending-send management, and
// communication thread lifecycle. Subclasses implement lookup() and
// communicationThreadMain().
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_MPISTORAGECACHE_H
#define PROTEUS_MPISTORAGECACHE_H

#include "proteus/impl/Caching/MPIHelpers.h"
#include "proteus/impl/Caching/ObjectCache.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Hashing.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace proteus {

class MPIStorageCache : public ObjectCache {
public:
  MPIStorageCache(const std::string &Label, int StoreTag);
  ~MPIStorageCache() override;

  void startCommThread();
  void store(const HashT &HashValue, const CacheEntry &Entry) override;
  void finalize() override;
  void printStats() override;
  uint64_t getHits() const override { return Hits; }
  uint64_t getAccesses() const override { return Accesses; }

protected:
  std::unique_ptr<CompiledLibrary> lookupFromDisk(const HashT &HashValue);
  virtual void communicationThreadMain() = 0;

  void forwardToWriter(const HashT &HashValue, const CacheEntry &Entry);
  void saveToDisk(const HashT &HashValue, const char *Data, size_t Size,
                  bool IsDynLib);

  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory;
  const std::string Label;
  const int StoreTag;
  MPICommHandle CommHandle;
  CommThreadHandle CommThread;

private:
  void pollPendingSends();
  void completeAllPendingSends();

  bool Finalized = false;
  std::vector<std::unique_ptr<PendingSend>> PendingSends;
};

} // namespace proteus

#endif
