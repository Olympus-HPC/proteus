//===-- MPIStorageCache.cpp -- MPI storage cache base class impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/Caching/MPIStorageCache.h"

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

static int mpiCleanupCallback(MPI_Comm, int, void *Attr, void *) {
  static_cast<MPIStorageCache *>(Attr)->finalize();
  return MPI_SUCCESS;
}

MPIStorageCache::MPIStorageCache(const std::string &Label)
    : StorageDirectory(Config::get().ProteusCacheDir
                           ? Config::get().ProteusCacheDir.value()
                           : ".proteus"),
      Label(Label) {
  std::filesystem::create_directories(StorageDirectory);

  int Keyval = MPI_KEYVAL_INVALID;
  MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, mpiCleanupCallback, &Keyval,
                         nullptr);
  MPI_Comm_set_attr(MPI_COMM_SELF, Keyval, this);
  MPI_Comm_free_keyval(&Keyval);
}

MPIStorageCache::~MPIStorageCache() = default;

void MPIStorageCache::finalize() {
  if (Finalized)
    return;

  int MPIFinalized = 0;
  MPI_Finalized(&MPIFinalized);
  if (MPIFinalized) {
    reportFatalError("[" + getName() +
                     "] MPI already finalized before cache finalized. This "
                     "should never happen");
  }

  if (Config::get().traceSpecializations()) {
    Logger::trace("[" + getName() + ":" + Label + "] Rank " +
                  std::to_string(CommHandle.getRank()) +
                  " flushing, PendingSends=" +
                  std::to_string(PendingSends.size()) + "\n");
  }

  MPI_Comm Comm = CommHandle.get();
  int Rank = CommHandle.getRank();

  completeAllPendingSends();

  // Only non-zero ranks send shutdown to the comm thread of rank 0.
  if (Rank != 0)
    MPI_Ssend(nullptr, 0, MPI_BYTE, 0, static_cast<int>(MPITag::Shutdown),
              Comm);

  CommThread.join();

  CommHandle.free();
  Finalized = true;
}

void MPIStorageCache::store(const HashT &HashValue, const CacheEntry &Entry) {
  TIMESCOPE(getName() + "::store");

  // Rank 0 main thread directly saves to disk, other ranks forward to rank 0's
  // communication thread.
  if (CommHandle.getRank() == 0) {
    saveToDisk(HashValue, Entry.Buffer.getBufferStart(),
               Entry.Buffer.getBufferSize(), Entry.IsDynLib);
    return;
  }
  forwardToWriter(HashValue, Entry);
}

void MPIStorageCache::startCommThread() {
  if (CommHandle.getRank() != 0)
    return;
  CommThread.start([this] { communicationThreadMain(); });
}

void MPIStorageCache::forwardToWriter(const HashT &HashValue,
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
                      /*dest=*/0, static_cast<int>(MPITag::Store), Comm,
                      &Pending->Request);
  if (Err != MPI_SUCCESS) {
    reportFatalError("MPI_Isend failed with error code " + std::to_string(Err));
  }

  PendingSends.push_back(std::move(Pending));
  pollPendingSends();
}

void MPIStorageCache::pollPendingSends() {
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

void MPIStorageCache::completeAllPendingSends() {
  for (auto &Pending : PendingSends) {
    MPI_Wait(&Pending->Request, MPI_STATUS_IGNORE);
  }
  PendingSends.clear();
}

void MPIStorageCache::communicationThreadMain() {
  if (Config::get().traceSpecializations()) {
    Logger::trace("[" + getName() + ":" + Label +
                  "] Communication thread started\n");
  }

  MPI_Comm Comm = CommHandle.get();
  int Size = CommHandle.getSize();

  // No rank to receive from, so exit.
  if (Size <= 1)
    return;

  int ShutdownCount = 0;

  try {
    while (true) {
      MPI_Status Status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, Comm, &Status);
      if (Status.MPI_SOURCE == 0)
        reportFatalError("Rank 0 should not receive messages on its own "
                         "communication thread.");

      auto Tag = static_cast<MPITag>(Status.MPI_TAG);

      if (Tag == MPITag::Shutdown) {
        MPI_Recv(nullptr, 0, MPI_BYTE, Status.MPI_SOURCE,
                 static_cast<int>(MPITag::Shutdown), Comm, MPI_STATUS_IGNORE);
        ShutdownCount++;
        // Check all other ranks have sent shutdown to exit.
        if (ShutdownCount >= (Size - 1))
          break;
      } else {
        handleMessage(Status, Tag);
      }
    }
  } catch (const std::exception &E) {
    reportFatalError(
        "[" + getName() +
        "] Communication thread encountered an exception: " + E.what());
  }

  if (Config::get().traceSpecializations()) {
    Logger::trace("[" + getName() + ":" + Label +
                  "] Communication thread exiting\n");
  }
}

void MPIStorageCache::handleMessage(MPI_Status &Status, MPITag Tag) {
  if (Tag == MPITag::Store)
    handleStoreMessage(Status);
  else
    reportFatalError("[" + getName() + "] Unexpected MPI tag: " +
                     std::to_string(static_cast<int>(Tag)));
}

void MPIStorageCache::handleStoreMessage(MPI_Status &Status) {
  MPI_Comm Comm = CommHandle.get();
  int MsgSize = 0;
  MPI_Get_count(&Status, MPI_BYTE, &MsgSize);

  std::vector<char> Buffer(MsgSize);
  MPI_Recv(Buffer.data(), MsgSize, MPI_BYTE, Status.MPI_SOURCE,
           static_cast<int>(MPITag::Store), Comm, MPI_STATUS_IGNORE);

  auto Msg = unpackStoreMessage(Comm, Buffer);
  saveToDisk(Msg.Hash, Msg.Data.data(), Msg.Data.size(), Msg.IsDynLib);
}

void MPIStorageCache::saveToDisk(const HashT &HashValue, const char *Data,
                                 size_t Size, bool IsDynLib) {
  std::string Filebase =
      StorageDirectory + "/cache-jit-" + HashValue.toString();
  std::string Extension = IsDynLib ? ".so" : ".o";
  std::string Filepath = Filebase + Extension;

  if (std::filesystem::exists(Filepath))
    return;

  saveToFileAtomic(Filepath, StringRef{Data, Size});

  if (Config::get().traceSpecializations()) {
    Logger::trace("[" + getName() + "] Saved " + Filepath + " (" +
                  std::to_string(Size) + " bytes)\n");
  }
}

std::unique_ptr<CompiledLibrary>
MPIStorageCache::lookupFromDisk(const HashT &HashValue) {
  std::string Filebase =
      StorageDirectory + "/cache-jit-" + HashValue.toString();

  auto CacheBuf = MemoryBuffer::getFileAsStream(Filebase + ".o");
  if (CacheBuf) {
    return std::make_unique<CompiledLibrary>(std::move(*CacheBuf));
  }

  if (std::filesystem::exists(Filebase + ".so")) {
    return std::make_unique<CompiledLibrary>(Filebase + ".so");
  }

  return nullptr;
}

void MPIStorageCache::printStats() {
  printf("[proteus][%s] %s rank %d/%d hits %lu accesses %lu\n", Label.c_str(),
         getName().c_str(), CommHandle.getRank(), CommHandle.getSize(), Hits,
         Accesses);
}

} // namespace proteus
