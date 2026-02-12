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
}

MPIStorageCache::~MPIStorageCache() { finalize(); }

void MPIStorageCache::finalize() {
  if (Finalized)
    return;

  int MPIFinalized = 0;
  MPI_Finalized(&MPIFinalized);
  if (MPIFinalized) {
    reportFatalError("[" + getName() +
                     "] MPI already finalized before cache cleanup. Ensure "
                     "proteus::finalize() is called before MPI_Finalize().");
  }

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[" + getName() + ":" + Label + "] Rank " +
                  std::to_string(CommHandle.getRank()) +
                  " flushing, PendingSends=" +
                  std::to_string(PendingSends.size()) + "\n");
  }

  MPI_Comm Comm = CommHandle.get();

  completeAllPendingSends();

  MPI_Ssend(nullptr, 0, MPI_BYTE, 0, static_cast<int>(MPITag::Shutdown), Comm);

  CommThread.join();

  MPI_Barrier(Comm);

  Finalized = true;
}

void MPIStorageCache::store(const HashT &HashValue, const CacheEntry &Entry) {
  TIMESCOPE(getName() + "::store");

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

void MPIStorageCache::saveToDisk(const HashT &HashValue, const char *Data,
                                 size_t Size, bool IsDynLib) {
  std::string Filebase =
      StorageDirectory + "/cache-jit-" + HashValue.toString();
  std::string Extension = IsDynLib ? ".so" : ".o";
  std::string Filepath = Filebase + Extension;

  if (std::filesystem::exists(Filepath))
    return;

  saveToFileAtomic(Filepath, StringRef{Data, Size});

  if (Config::get().ProteusTraceOutput >= 1) {
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
