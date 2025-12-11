//===-- MPISharedStorageCache.cpp -- MPI shared storage cache impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <filesystem>
#include <limits>

#include <mpi.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>

#include "proteus/Caching/MPISharedStorageCache.h"
#include "proteus/Config.h"
#include "proteus/Error.h"
#include "proteus/Logger.h"
#include "proteus/TimeTracing.h"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;
namespace {
constexpr int MaxRequestsPerCall = 5;
} // namespace

struct UnpackedMessage {
  HashT Hash;
  std::vector<char> Data;
  bool IsDynLib;
};

static UnpackedMessage unpackMessage(const std::vector<char> &Buffer) {
  const char *Ptr = Buffer.data();

  uint32_t HashSize = 0;
  std::memcpy(&HashSize, Ptr, sizeof(HashSize));
  Ptr += sizeof(HashSize);

  std::string HashStr(Ptr, HashSize);
  Ptr += HashSize;

  uint8_t IsDynLib = 0;
  std::memcpy(&IsDynLib, Ptr, sizeof(IsDynLib));
  Ptr += sizeof(IsDynLib);

  uint64_t BufferSize = 0;
  std::memcpy(&BufferSize, Ptr, sizeof(BufferSize));
  Ptr += sizeof(BufferSize);

  std::vector<char> Data(BufferSize);
  std::memcpy(Data.data(), Ptr, BufferSize);

  return UnpackedMessage{HashT(StringRef(HashStr)), std::move(Data),
                         IsDynLib != 0};
}

//===----------------------------------------------------------------------===//
// MPICommHandle implementation
//===----------------------------------------------------------------------===//

MPICommHandle::~MPICommHandle() {
  if (Comm == MPI_COMM_NULL)
    return;

  int MPIFinalized = 0;
  MPI_Finalized(&MPIFinalized);
  if (!MPIFinalized)
    MPI_Comm_free(&Comm);
}

void MPICommHandle::ensureInitialized() {
  if (Comm != MPI_COMM_NULL)
    return;

  int MPIInitialized = 0;
  MPI_Initialized(&MPIInitialized);
  if (!MPIInitialized) {
    reportFatalError("MPICommHandle requires MPI to be initialized");
  }

  MPI_Comm_dup(MPI_COMM_WORLD, &Comm);
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(Comm, &Size);

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPICommHandle] Initialized communicator for rank " +
                  std::to_string(Rank) + "/" + std::to_string(Size) + "\n");
  }
}

MPI_Comm MPICommHandle::get() {
  ensureInitialized();
    return Comm;
}

int MPICommHandle::getRank() {
  ensureInitialized();
  return Rank;
}

int MPICommHandle::getSize() {
  ensureInitialized();
  return Size;
}

//===----------------------------------------------------------------------===//
// MPISharedStorageCache implementation
//===----------------------------------------------------------------------===//

int MPISharedStorageCache::computeTag(const std::string &Label) {
  return 0;
}

MPISharedStorageCache::MPISharedStorageCache(const std::string &Label)
    : StorageDirectory(Config::get().ProteusCacheDir
                           ? Config::get().ProteusCacheDir.value()
                           : ".proteus"),
      Label(Label), Tag(computeTag(Label)) {
  std::filesystem::create_directory(StorageDirectory);
}

MPISharedStorageCache::~MPISharedStorageCache() { flush(); }

void MPISharedStorageCache::flush() {
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
  MPI_Barrier(Comm);

  if (CommHandle.getRank() == 0)
    receiveIncoming(std::numeric_limits<int>::max());
  else
    waitForPendingSends();

  MPI_Barrier(Comm);

  Finalized = true;
}

std::unique_ptr<CompiledLibrary>
MPISharedStorageCache::lookup(const HashT &HashValue) {
  TIMESCOPE("MPISharedStorageCache::lookup");
  Accesses++;

  if (CommHandle.getRank() == 0)
    receiveIncoming(MaxRequestsPerCall);

  std::string Filebase =
      StorageDirectory + "/cache-jit-" + HashValue.toString();

  auto CacheBuf = MemoryBuffer::getFileAsStream(Filebase + ".o");
  if (CacheBuf) {
    Hits++;
    return std::make_unique<CompiledLibrary>(std::move(*CacheBuf));
  }

  if (std::filesystem::exists(Filebase + ".so")) {
    Hits++;
    return std::make_unique<CompiledLibrary>(
        SmallString<128>{Filebase + ".so"});
  }

  return nullptr;
}

void MPISharedStorageCache::store(const HashT &HashValue, const CacheEntry &Entry) {
  TIMESCOPE("MPISharedStorageCache::store");

  if (CommHandle.getRank() == 0) {
    receiveIncoming(MaxRequestsPerCall);
    saveToDisk(HashValue, Entry.Buffer.getBufferStart(),
               Entry.Buffer.getBufferSize(), Entry.isSharedObject());
  } else {
    forwardToWriter(HashValue, Entry);
  }
}

void MPISharedStorageCache::receiveIncoming(int MaxMessages) {
  int Flag = 0;
  MPI_Status Status;
  int Received = 0;
  MPI_Comm Comm = CommHandle.get();

  while (Received < MaxMessages) {
    MPI_Iprobe(MPI_ANY_SOURCE, Tag, Comm, &Flag, &Status);
    if (!Flag)
      break;

    int MsgSize = 0;
    MPI_Get_count(&Status, MPI_BYTE, &MsgSize);

    std::vector<char> Buffer(MsgSize);
    MPI_Recv(Buffer.data(), MsgSize, MPI_BYTE, Status.MPI_SOURCE, Tag, Comm,
             MPI_STATUS_IGNORE);

    auto [Hash, Data, IsDynLib] = unpackMessage(Buffer);
    saveToDisk(Hash, Data.data(), Data.size(), IsDynLib);

    ++Received;
  }
}

void MPISharedStorageCache::forwardToWriter(const HashT &HashValue,
                                            const CacheEntry &Entry) {
  auto Pending = std::make_unique<PendingSend>();
  Pending->Buffer = packMessage(HashValue, Entry);

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
    reportFatalError("MPI_Isend failed with error code " +
                        std::to_string(Err));
  }

  PendingSends.push_back(std::move(Pending));
}

void MPISharedStorageCache::waitForPendingSends() {
  for (auto &Pending : PendingSends) {
    MPI_Wait(&Pending->Request, MPI_STATUS_IGNORE);
  }
  PendingSends.clear();
}

std::vector<char> MPISharedStorageCache::packMessage(const HashT &HashValue,
                                                     const CacheEntry &Entry) {
  // Format: [hash_size (4 bytes), hash_bytes, is_dynlib (1 byte),
  //          buffer_size (8 bytes), buffer_bytes]
  std::string HashStr = HashValue.toString();
  uint32_t HashSize = static_cast<uint32_t>(HashStr.size());
  uint8_t IsDynLib = Entry.isSharedObject() ? 1 : 0;
  uint64_t BufferSize = Entry.Buffer.getBufferSize();

  size_t TotalSize = sizeof(HashSize) + HashSize + sizeof(IsDynLib) +
                     sizeof(BufferSize) + BufferSize;
  std::vector<char> Packed(TotalSize);
  char *Ptr = Packed.data();

  std::memcpy(Ptr, &HashSize, sizeof(HashSize));
  Ptr += sizeof(HashSize);

  std::memcpy(Ptr, HashStr.data(), HashSize);
  Ptr += HashSize;

  std::memcpy(Ptr, &IsDynLib, sizeof(IsDynLib));
  Ptr += sizeof(IsDynLib);

  std::memcpy(Ptr, &BufferSize, sizeof(BufferSize));
  Ptr += sizeof(BufferSize);

  std::memcpy(Ptr, Entry.Buffer.getBufferStart(), BufferSize);

  return Packed;
}

void MPISharedStorageCache::saveToDisk(const HashT &HashValue, const char *Data,
                                       size_t Size, bool IsDynLib) {
  std::string Filebase =
      StorageDirectory + "/cache-jit-" + HashValue.toString();
  std::string Extension = IsDynLib ? ".so" : ".o";
  std::string Filepath = Filebase + Extension;

  // Skip writing if this hash is already cached.
  if (std::filesystem::exists(Filepath))
    return;

  saveToFile(Filepath, StringRef{Data, Size});

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
