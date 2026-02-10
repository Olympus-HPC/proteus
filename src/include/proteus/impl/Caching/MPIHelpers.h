//===-- MPIHelpers.h -- MPI helper classes for cache implementations --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared MPI helper classes used by MPI-based cache implementations.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_MPIHELPERS_H
#define PROTEUS_MPIHELPERS_H

#include "proteus/impl/Caching/ObjectCache.h"
#include "proteus/impl/Hashing.h"

#include <mpi.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace proteus {

void validateMPIConfig();

/// Manages the lifecycle of a background communication thread.
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

/// RAII wrapper for MPI communicator with thread safety checks.
class MPICommHandle {
public:
  MPICommHandle();
  ~MPICommHandle();

  MPICommHandle(const MPICommHandle &) = delete;
  MPICommHandle &operator=(const MPICommHandle &) = delete;

  MPI_Comm get() const;
  int getRank() const;
  int getSize() const;

private:
  MPI_Comm Comm = MPI_COMM_NULL;
  int Rank = -1;
  int Size = -1;
};

/// Tracks an asynchronous MPI send operation.
struct PendingSend {
  std::vector<char> Buffer;
  MPI_Request Request;
};

/// Wire-format message for store operations: [HashSize, HashStr, IsDynLib,
/// DataSize, Data].
struct StoreMessage {
  HashT Hash;
  std::vector<char> Data;
  bool IsDynLib;
};

std::vector<char> packStoreMessage(MPI_Comm Comm, const HashT &HashValue,
                                   const CacheEntry &Entry);
StoreMessage unpackStoreMessage(MPI_Comm Comm, const std::vector<char> &Buffer);

} // namespace proteus

#endif
