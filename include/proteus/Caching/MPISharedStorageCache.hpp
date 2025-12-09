//===-- MPISharedStorageCache.hpp -- MPI shared storage cache header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_MPISHAREDSTORAGECACHE_HPP
#define PROTEUS_MPISHAREDSTORAGECACHE_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#include "proteus/Caching/ObjectCache.hpp"
#include "proteus/CompiledLibrary.hpp"
#include "proteus/Hashing.hpp"

namespace proteus {

class MPICommHandle {
public:
  MPICommHandle() = default;
  ~MPICommHandle();

  MPICommHandle(const MPICommHandle &) = delete;
  MPICommHandle &operator=(const MPICommHandle &) = delete;

  void set(MPI_Comm Comm);
  MPI_Comm get() const;
  void reset();
  bool isSet() const { return Comm != MPI_COMM_NULL; }

private:
  MPI_Comm Comm = MPI_COMM_NULL;
  bool Owned = false;
};

struct PendingSend {
  std::vector<char> Buffer;
  MPI_Request Request;
};

class MPISharedStorageCache : public ObjectCache {
public:
  MPISharedStorageCache(const std::string &Label, MPI_Comm Comm);
  ~MPISharedStorageCache() override;

  std::string getName() const override { return "MPISharedStorage"; }

  std::unique_ptr<CompiledLibrary> lookup(HashT &HashValue) override;

  void store(HashT &HashValue, const CacheEntry &Entry) override;

  void flush() override;

  void printStats() override;

  uint64_t getHits() const override { return Hits; }

  uint64_t getAccesses() const override { return Accesses; }

  static void setDefaultCommunicator(MPI_Comm Comm);
  static MPI_Comm getDefaultCommunicator();
  static void clearDefaultCommunicator();

private:
  void receiveIncoming(int MaxMessages);
  void forwardToWriter(HashT &HashValue, const CacheEntry &Entry);
  void waitForPendingSends();
  std::vector<char> packMessage(const HashT &HashValue,
                                const CacheEntry &Entry);
  void saveToDisk(const HashT &HashValue, const char *Data, size_t Size,
                  bool IsDynLib);
  static int computeTag(const std::string &Label);

  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory;
  const std::string Label;
  const int Tag;
  MPICommHandle CommHandle;
  int Rank = 0;
  int Size = 1;
  bool IsWriter = true;
  bool Finalized = false;
  std::vector<std::unique_ptr<PendingSend>> PendingSends;
};

} // namespace proteus

#endif
