//===-- MPISharedStorageCache.h -- MPI shared storage cache header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_MPISHAREDSTORAGECACHE_H
#define PROTEUS_MPISHAREDSTORAGECACHE_H

#include "proteus/impl/Caching/MPIHelpers.h"
#include "proteus/impl/Caching/ObjectCache.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Hashing.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace proteus {

class MPISharedStorageCache : public ObjectCache {
public:
  MPISharedStorageCache(const std::string &Label);
  ~MPISharedStorageCache() override;

  std::string getName() const override { return "MPISharedStorage"; }

  std::unique_ptr<CompiledLibrary> lookup(const HashT &HashValue) override;

  void store(const HashT &HashValue, const CacheEntry &Entry) override;

  void finalize() override;

  void printStats() override;

  uint64_t getHits() const override { return Hits; }

  uint64_t getAccesses() const override { return Accesses; }

private:
  void forwardToWriter(const HashT &HashValue, const CacheEntry &Entry);
  void pollPendingSends();
  void completeAllPendingSends();
  void saveToDisk(const HashT &HashValue, const char *Data, size_t Size,
                  bool IsDynLib);
  static int computeTag(const std::string &Label);

  void communicationThreadMain();
  void startCommThread();

  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory;
  const std::string Label;
  const int Tag;
  MPICommHandle CommHandle;
  CommThreadHandle CommThread;
  bool Finalized = false;
  std::vector<std::unique_ptr<PendingSend>> PendingSends;
};

} // namespace proteus

#endif
