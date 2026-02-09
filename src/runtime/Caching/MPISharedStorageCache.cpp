//===-- MPISharedStorageCache.cpp -- MPI shared storage cache impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/Caching/MPISharedStorageCache.h"

#include "proteus/Error.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/TimeTracing.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>

namespace proteus {

using namespace llvm;

//===----------------------------------------------------------------------===//
// MPISharedStorageCache implementation
//===----------------------------------------------------------------------===//

int MPISharedStorageCache::computeTag(const std::string &Label) { return 0; }

MPISharedStorageCache::MPISharedStorageCache(const std::string &Label)
    : StorageDirectory(Config::get().ProteusCacheDir
                           ? Config::get().ProteusCacheDir.value()
                           : ".proteus"),
      Label(Label), Tag(computeTag(Label)) {
  std::filesystem::create_directories(StorageDirectory);
  startCommThread();
}

MPISharedStorageCache::~MPISharedStorageCache() { finalize(); }

void MPISharedStorageCache::finalize() {
  if (Finalized)
    return;

  int MPIFinalized = 0;
  MPI_Finalized(&MPIFinalized);
  if (MPIFinalized) {
    if (!PendingSends.empty()) {
      Logger::trace("[MPISharedStorageCache] Warning: MPI already finalized, "
                    "cannot complete " +
                    std::to_string(PendingSends.size()) + " pending sends\n");
    }
    CommThread.stop();
    Finalized = true;
    return;
  }

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPISharedStorageCache:" + Label + "] Rank " +
                  std::to_string(CommHandle.getRank()) +
                  " flushing, PendingSends=" +
                  std::to_string(PendingSends.size()) + "\n");
  }

  MPI_Comm Comm = CommHandle.get();

  completeAllPendingSends();

  MPI_Barrier(Comm);

  // Phase 2: Stop the communication thread (rank 0).
  // Thread drains remaining messages before exiting.
  CommThread.stop();

  MPI_Barrier(Comm);

  Finalized = true;
}

std::unique_ptr<CompiledLibrary>
MPISharedStorageCache::lookup(const HashT &HashValue) {
  TIMESCOPE("MPISharedStorageCache::lookup");
  Accesses++;

  std::string Filebase =
      StorageDirectory + "/cache-jit-" + HashValue.toString();

  auto CacheBuf = MemoryBuffer::getFileAsStream(Filebase + ".o");
  if (CacheBuf) {
    Hits++;
    return std::make_unique<CompiledLibrary>(std::move(*CacheBuf));
  }

  if (std::filesystem::exists(Filebase + ".so")) {
    Hits++;
    return std::make_unique<CompiledLibrary>(Filebase + ".so");
  }

  return nullptr;
}

void MPISharedStorageCache::store(const HashT &HashValue,
                                  const CacheEntry &Entry) {
  TIMESCOPE("MPISharedStorageCache::store");

  forwardToWriter(HashValue, Entry);
}

void MPISharedStorageCache::communicationThreadMain() {
  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPISharedStorageCache:" + Label +
                  "] Communication thread started\n");
  }

  MPI_Comm Comm = CommHandle.get();

  while (true) {
    int Flag = 0;
    MPI_Status Status;
    MPI_Iprobe(MPI_ANY_SOURCE, Tag, Comm, &Flag, &Status);

    if (Flag) {
      int MsgSize = 0;
      MPI_Get_count(&Status, MPI_BYTE, &MsgSize);

      std::vector<char> Buffer(MsgSize);
      MPI_Recv(Buffer.data(), MsgSize, MPI_BYTE, Status.MPI_SOURCE, Tag, Comm,
               MPI_STATUS_IGNORE);

      auto Msg = unpackStoreMessage(Comm, Buffer);
      saveToDisk(Msg.Hash, Msg.Data.data(), Msg.Data.size(), Msg.IsDynLib);
    } else {
      if (CommThread.shutdownRequested()) {
        // Final drain: one more probe to ensure queue is empty.
        MPI_Iprobe(MPI_ANY_SOURCE, Tag, Comm, &Flag, &Status);
        if (!Flag)
          break;
      } else {
        CommThread.waitOrShutdown(std::chrono::milliseconds(1));
      }
    }
  }

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPISharedStorageCache:" + Label +
                  "] Communication thread exiting\n");
  }
}

void MPISharedStorageCache::startCommThread() {
  if (CommHandle.getRank() != 0)
    return;
  CommThread.start([this] { communicationThreadMain(); });
}

void MPISharedStorageCache::forwardToWriter(const HashT &HashValue,
                                            const CacheEntry &Entry) {
  auto Pending = std::make_unique<PendingSend>();
  Pending->Buffer = packStoreMessage(CommHandle.get(), HashValue, Entry);

  if (Pending->Buffer.size() >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    reportFatalError("MPI message size exceeds INT_MAX: " +
                     std::to_string(Pending->Buffer.size()) + " bytes");
  }

  MPI_Comm Comm = CommHandle.get();
  int Err = MPI_Isend(Pending->Buffer.data(),
                      static_cast<int>(Pending->Buffer.size()), MPI_BYTE,
                      /*dest=*/0, Tag, Comm, &Pending->Request);
  if (Err != MPI_SUCCESS) {
    reportFatalError("MPI_Isend failed with error code " + std::to_string(Err));
  }

  PendingSends.push_back(std::move(Pending));
  pollPendingSends();
}

void MPISharedStorageCache::pollPendingSends() {
  if (PendingSends.empty())
    return;

  auto IsDone = [](const std::unique_ptr<PendingSend> &Pending) {
    int Done = 0;
    MPI_Test(&Pending->Request, &Done, MPI_STATUS_IGNORE);
    return Done != 0;
  };

  PendingSends.erase(
      std::remove_if(PendingSends.begin(), PendingSends.end(), IsDone),
      PendingSends.end());
}

void MPISharedStorageCache::completeAllPendingSends() {
  for (auto &Pending : PendingSends) {
    MPI_Wait(&Pending->Request, MPI_STATUS_IGNORE);
  }
  PendingSends.clear();
}

void MPISharedStorageCache::saveToDisk(const HashT &HashValue, const char *Data,
                                       size_t Size, bool IsDynLib) {
  std::string Filebase =
      StorageDirectory + "/cache-jit-" + HashValue.toString();
  std::string Extension = IsDynLib ? ".so" : ".o";
  std::string Filepath = Filebase + Extension;

  if (std::filesystem::exists(Filepath))
    return;

  saveToFileAtomic(Filepath, StringRef{Data, Size});

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPISharedStorageCache] Saved " + Filebase + Extension +
                  " (" + std::to_string(Size) + " bytes)\n");
  }
}

void MPISharedStorageCache::printStats() {
  printf("[proteus][%s] MPISharedStorageCache rank %d/%d hits %lu accesses "
         "%lu\n",
         Label.c_str(), CommHandle.getRank(), CommHandle.getSize(), Hits,
         Accesses);
}

} // namespace proteus
