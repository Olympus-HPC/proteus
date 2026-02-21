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

#include "proteus/Error.h"
#include "proteus/impl/Caching/ObjectCache.h"
#include "proteus/impl/Hashing.h"

#include <mpi.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

// Check an MPI call's return code. On failure, format a human-readable error
// message using MPI_Error_string and report it as a fatal error.
#define proteusMpiCheck(CALL)                                                  \
  do {                                                                         \
    int proteus_mpi_err_ = (CALL);                                             \
    if (proteus_mpi_err_ != MPI_SUCCESS) {                                     \
      char proteus_mpi_str_[MPI_MAX_ERROR_STRING];                             \
      int proteus_mpi_len_ = 0;                                                \
      MPI_Error_string(proteus_mpi_err_, proteus_mpi_str_, &proteus_mpi_len_); \
      std::string proteus_mpi_msg_ =                                           \
          std::string(#CALL) +                                                 \
          " failed: " + std::string(proteus_mpi_str_, proteus_mpi_len_) +      \
          " at " + __FILE__ + ":" + std::to_string(__LINE__);                  \
      reportFatalError(proteus_mpi_msg_);                                      \
    }                                                                          \
  } while (0)

namespace proteus {

enum class MPITag : int {
  Store = 1,
  LookupRequest = 2,
  LookupResponse = 3,
  Shutdown = 4
};

void validateMPIConfig();

/// Manages the lifecycle of a background communication thread.
class CommThreadHandle {
public:
  CommThreadHandle() = default;
  ~CommThreadHandle();

  CommThreadHandle(const CommThreadHandle &) = delete;
  CommThreadHandle &operator=(const CommThreadHandle &) = delete;

  template <typename Callable> void start(Callable &&ThreadFunc) {
    if (isRunning())
      return;

    Thread = std::make_unique<std::thread>(std::forward<Callable>(ThreadFunc));
    Running = true;
  }

  void join();

  bool isRunning() const;

private:
  std::unique_ptr<std::thread> Thread;
  bool Running = false;
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

  void free();

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
