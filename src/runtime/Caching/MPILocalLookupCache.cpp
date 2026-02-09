//===-- MPILocalLookupCache.cpp -- MPI local-lookup cache impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/Caching/MPILocalLookupCache.h"

#include "proteus/impl/Config.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/TimeTracing.h"

#include <mpi.h>

namespace proteus {

MPILocalLookupCache::MPILocalLookupCache(const std::string &Label)
    : MPIStorageCache(Label, /*StoreTag=*/0) {}

std::unique_ptr<CompiledLibrary>
MPILocalLookupCache::lookup(const HashT &HashValue) {
  TIMESCOPE("MPILocalLookupCache::lookup");
  Accesses++;

  auto Result = lookupFromDisk(HashValue);
  if (Result)
    Hits++;

  return Result;
}

void MPILocalLookupCache::communicationThreadMain() {
  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPILocalLookup:" + Label +
                  "] Communication thread started\n");
  }

  MPI_Comm Comm = CommHandle.get();

  while (true) {
    int Flag = 0;
    MPI_Status Status;
    MPI_Iprobe(MPI_ANY_SOURCE, StoreTag, Comm, &Flag, &Status);

    if (Flag) {
      int MsgSize = 0;
      MPI_Get_count(&Status, MPI_BYTE, &MsgSize);

      std::vector<char> Buffer(MsgSize);
      MPI_Recv(Buffer.data(), MsgSize, MPI_BYTE, Status.MPI_SOURCE, StoreTag,
               Comm, MPI_STATUS_IGNORE);

      auto Msg = unpackStoreMessage(Comm, Buffer);
      saveToDisk(Msg.Hash, Msg.Data.data(), Msg.Data.size(), Msg.IsDynLib);
    } else {
      if (CommThread.shutdownRequested()) {
        MPI_Iprobe(MPI_ANY_SOURCE, StoreTag, Comm, &Flag, &Status);
        if (!Flag)
          break;
      } else {
        CommThread.waitOrShutdown(std::chrono::milliseconds(1));
      }
    }
  }

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPILocalLookup:" + Label +
                  "] Communication thread exiting\n");
  }
}

} // namespace proteus
