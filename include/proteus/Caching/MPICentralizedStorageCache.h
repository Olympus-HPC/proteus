//===-- MPICentralizedStorageCache.h -- Centralized MPI cache header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A fully centralized MPI storage cache where Rank 0 is the single writer AND
// reader. Other ranks make synchronous MPI requests to Rank 0 for both store()
// and lookup() operations.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_MPICENTRALIZEDSTORAGECACHE_H
#define PROTEUS_MPICENTRALIZEDSTORAGECACHE_H

#include "proteus/Caching/MPIHelpers.h"
#include "proteus/Caching/ObjectCache.h"
#include "proteus/CompiledLibrary.h"
#include "proteus/Hashing.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace proteus {

struct LookupRequest;
struct LookupResponse;

class MPICentralizedStorageCache : public ObjectCache {
public:
  MPICentralizedStorageCache(const std::string &Label);
  ~MPICentralizedStorageCache() override;

  std::string getName() const override { return "MPICentralizedStorage"; }

  std::unique_ptr<CompiledLibrary> lookup(const HashT &HashValue) override;

  void store(const HashT &HashValue, const CacheEntry &Entry) override;

  void finalize() override;

  void printStats() override;

  uint64_t getHits() const override { return Hits; }

  uint64_t getAccesses() const override { return Accesses; }

private:
  std::unique_ptr<CompiledLibrary> lookupRemote(const HashT &HashValue);
  std::unique_ptr<CompiledLibrary> lookupLocal(const HashT &HashValue);

  void handleStoreMessage(MPI_Status &Status);
  void handleLookupRequest(MPI_Status &Status);

  std::vector<char> packLookupRequest(const HashT &HashValue);
  std::vector<char> packLookupResponse(bool Found, bool IsDynLib,
                                       const std::vector<char> &Data);
  LookupRequest unpackLookupRequest(const std::vector<char> &Buffer);
  LookupResponse unpackLookupResponse(const std::vector<char> &Buffer);

  void saveToDisk(const HashT &HashValue, const char *Data, size_t Size,
                  bool IsDynLib);

  void forwardToWriter(const HashT &HashValue, const CacheEntry &Entry);
  void pollPendingSends();
  void completeAllPendingSends();

  void communicationThreadMain();
  void startCommThread();

  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory;
  const std::string Label;
  MPICommHandle CommHandle;
  CommThreadHandle CommThread;
  bool Finalized = false;
  std::vector<std::unique_ptr<PendingSend>> PendingSends;

  static constexpr int TagStore = 0;
  static constexpr int TagLookupRequest = 1;
  static constexpr int TagLookupResponse = 2;
};

} // namespace proteus

#endif
