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

#include "proteus/Error.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/TimeTracing.h"

#include <mpi.h>

namespace proteus {

MPILocalLookupCache::MPILocalLookupCache(const std::string &Label)
    : MPIStorageCache(Label) {
  startCommThread();
}

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
  int Size = CommHandle.getSize();
  int ShutdownCount = 0;

  try {
    while (true) {
      MPI_Status Status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, Comm, &Status);

      auto Tag = static_cast<MPITag>(Status.MPI_TAG);

      if (Tag == MPITag::Shutdown) {
        MPI_Recv(nullptr, 0, MPI_BYTE, Status.MPI_SOURCE,
                 static_cast<int>(MPITag::Shutdown), Comm, MPI_STATUS_IGNORE);
        ++ShutdownCount;
        if (ShutdownCount == Size)
          break;
      } else if (Tag == MPITag::Store) {
        handleStoreMessage(Status);
      } else {
        reportFatalError("[MPILocalLookup] Unexpected MPI tag: " +
                         std::to_string(static_cast<int>(Tag)));
      }
    }
  } catch (const std::exception &E) {
    reportFatalError(std::string("[MPILocalLookup] Communication thread "
                                 "encountered an exception: ") +
                     E.what());
  }

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPILocalLookup:" + Label +
                  "] Communication thread exiting\n");
  }
}

} // namespace proteus
