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

#include "proteus/impl/Caching/ObjectCache.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Hashing.h"

#include <mpi.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace proteus {

void validateMPIConfig();

class CommThreadHandle {
public:
  CommThreadHandle() = default;
  ~CommThreadHandle();

  CommThreadHandle(const CommThreadHandle &) = delete;
  CommThreadHandle &operator=(const CommThreadHandle &) = delete;

  template <typename Callable> void start(Callable &&ThreadFunc) {
    if (Running.load())
      return;

    ShutdownFlag.store(false, std::memory_order_release);
    Thread = std::make_unique<std::thread>(std::forward<Callable>(ThreadFunc));
    Running.store(true, std::memory_order_release);
  }

  void stop();

  bool isRunning() const;

  bool shutdownRequested() const;

  bool waitOrShutdown(std::chrono::milliseconds Timeout);

private:
  std::unique_ptr<std::thread> Thread;
  mutable std::mutex Mutex;
  std::condition_variable CondVar;
  std::atomic<bool> ShutdownFlag{false};
  std::atomic<bool> Running{false};
};

class MPICommHandle {
public:
  MPICommHandle();
  ~MPICommHandle();

  MPICommHandle(const MPICommHandle &) = delete;
  MPICommHandle &operator=(const MPICommHandle &) = delete;

  MPI_Comm get();
  int getRank();
  int getSize();

private:
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

  void finalize() override;

  void printStats() override;

  uint64_t getHits() const override { return Hits; }

  uint64_t getAccesses() const override { return Accesses; }

private:
  void forwardToWriter(const HashT &HashValue, const CacheEntry &Entry);
  void pollPendingSends();
  void completeAllPendingSends();
  std::vector<char> packMessage(const HashT &HashValue,
                                const CacheEntry &Entry);
  UnpackedMessage unpackMessage(const std::vector<char> &Buffer);
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
  const int ShutdownTag;
  const int AckTag;
  MPICommHandle CommHandle;
  CommThreadHandle CommThread;
  bool Finalized = false;
  std::vector<std::unique_ptr<PendingSend>> PendingSends;
};

} // namespace proteus

#endif
