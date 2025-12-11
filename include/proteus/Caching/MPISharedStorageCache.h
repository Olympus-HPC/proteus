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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#include "proteus/Caching/ObjectCache.h"
#include "proteus/CompiledLibrary.h"
#include "proteus/Hashing.h"

namespace proteus {

class MPICommHandle {
public:
  MPICommHandle() = default;
  ~MPICommHandle();

  MPICommHandle(const MPICommHandle &) = delete;
  MPICommHandle &operator=(const MPICommHandle &) = delete;

  MPI_Comm get();
  int getRank();
  int getSize();

private:
  void ensureInitialized();

  MPI_Comm Comm = MPI_COMM_NULL;
  int Rank = -1;
  int Size = -1;
};

struct PendingSend {
  std::vector<char> Buffer;
  MPI_Request Request;
};

struct UnpackedMessage {
  HashT Hash;
  std::vector<char> Data;
  bool IsDynLib;
};

class MPISharedStorageCache : public ObjectCache {
public:
  MPISharedStorageCache(const std::string &Label);
  ~MPISharedStorageCache() override;

  std::string getName() const override { return "MPISharedStorage"; }

  std::unique_ptr<CompiledLibrary> lookup(const HashT &HashValue) override;

  void store(const HashT &HashValue, const CacheEntry &Entry) override;

  void flush() override;

  void printStats() override;

  uint64_t getHits() const override { return Hits; }

  uint64_t getAccesses() const override { return Accesses; }

private:
  void receiveIncoming(int MaxMessages);
  void forwardToWriter(const HashT &HashValue, const CacheEntry &Entry);
  void waitForPendingSends();
  std::vector<char> packMessage(const HashT &HashValue,
                                const CacheEntry &Entry);
  UnpackedMessage unpackMessage(const std::vector<char> &Buffer);
  void saveToDisk(const HashT &HashValue, const char *Data, size_t Size,
                  bool IsDynLib);
  static int computeTag(const std::string &Label);

  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory;
  const std::string Label;
  const int Tag;
  MPICommHandle CommHandle;
  bool Finalized = false;
  std::vector<std::unique_ptr<PendingSend>> PendingSends;
};

} // namespace proteus

#endif
