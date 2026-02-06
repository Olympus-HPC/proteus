//===-- MPISharedStorageCache.cpp -- MPI shared storage cache impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/Caching/MPISharedStorageCache.h"

#include "proteus/Config.h"
#include "proteus/Error.h"
#include "proteus/Logger.h"
#include "proteus/TimeTracing.h"
#include "proteus/Utils.h"

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
// MPI Validation
//===----------------------------------------------------------------------===//

void validateMPIForProteus() {
  int MPIInitialized = 0;
  MPI_Initialized(&MPIInitialized);
  if (!MPIInitialized) {
    reportFatalError("proteus::init() with mpi-storage cache requires MPI to "
                     "be initialized. Call MPI_Init_thread() before "
                     "proteus::init()");
  }

  int Provided = 0;
  MPI_Query_thread(&Provided);
  if (Provided != MPI_THREAD_MULTIPLE) {
    reportFatalError("MPISharedStorageCache requires MPI_THREAD_MULTIPLE "
                     "(provided level: " +
                     std::to_string(Provided) +
                     "). Initialize MPI with MPI_Init_thread()");
  }
}

//===----------------------------------------------------------------------===//
// MPICommHandle implementation
//===----------------------------------------------------------------------===//

MPICommHandle::MPICommHandle() {
  validateMPIForProteus();

  MPI_Comm_dup(MPI_COMM_WORLD, &Comm);
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(Comm, &Size);

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPICommHandle] Initialized communicator for rank " +
                  std::to_string(Rank) + "/" + std::to_string(Size) + "\n");
  }
}

MPICommHandle::~MPICommHandle() {
  if (Comm == MPI_COMM_NULL)
    return;

  int MPIFinalized = 0;
  MPI_Finalized(&MPIFinalized);
  if (!MPIFinalized)
    MPI_Comm_free(&Comm);
}

//===----------------------------------------------------------------------===//
// CommThreadHandle implementation
//===----------------------------------------------------------------------===//

CommThreadHandle::~CommThreadHandle() { stop(); }

void CommThreadHandle::stop() {
  if (!Thread)
    return;

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    ShutdownFlag.store(true, std::memory_order_release);
  }
  CondVar.notify_all();

  if (Thread->joinable())
    Thread->join();
  Thread.reset();
  Running.store(false, std::memory_order_release);
}

bool CommThreadHandle::isRunning() const {
  return Running.load(std::memory_order_acquire);
}

bool CommThreadHandle::shutdownRequested() const {
  return ShutdownFlag.load(std::memory_order_acquire);
}

bool CommThreadHandle::waitOrShutdown(std::chrono::milliseconds Timeout) {
  std::unique_lock<std::mutex> Lock(Mutex);
  return CondVar.wait_for(Lock, Timeout,
                          [this] { return ShutdownFlag.load(); });
}

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

      auto Msg = unpackMessage(Buffer);
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

std::vector<char> MPISharedStorageCache::packMessage(const HashT &HashValue,
                                                     const CacheEntry &Entry) {
  MPI_Comm Comm = CommHandle.get();
  std::string HashStr = HashValue.toString();
  uint32_t HashSize = static_cast<uint32_t>(HashStr.size());
  uint8_t IsDynLib = Entry.isSharedObject() ? 1 : 0;
  uint64_t BufferSize = Entry.Buffer.getBufferSize();

  if (BufferSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
    reportFatalError("Buffer size exceeds MPI int limit: " +
                     std::to_string(BufferSize) + " bytes");
  }

  int HashSizeBytes, HashStrBytes, FlagBytes, BufSizeBytes, DataBytes;
  MPI_Pack_size(1, MPI_UINT32_T, Comm, &HashSizeBytes);
  MPI_Pack_size(static_cast<int>(HashSize), MPI_CHAR, Comm, &HashStrBytes);
  MPI_Pack_size(1, MPI_BYTE, Comm, &FlagBytes);
  MPI_Pack_size(1, MPI_UINT64_T, Comm, &BufSizeBytes);
  MPI_Pack_size(static_cast<int>(BufferSize), MPI_BYTE, Comm, &DataBytes);

  int TotalSize =
      HashSizeBytes + HashStrBytes + FlagBytes + BufSizeBytes + DataBytes;
  std::vector<char> Packed(TotalSize);
  int Position = 0;

  MPI_Pack(&HashSize, 1, MPI_UINT32_T, Packed.data(), TotalSize, &Position,
           Comm);

  MPI_Pack(HashStr.data(), static_cast<int>(HashSize), MPI_CHAR, Packed.data(),
           TotalSize, &Position, Comm);

  MPI_Pack(&IsDynLib, 1, MPI_BYTE, Packed.data(), TotalSize, &Position, Comm);

  MPI_Pack(&BufferSize, 1, MPI_UINT64_T, Packed.data(), TotalSize, &Position,
           Comm);

  MPI_Pack(const_cast<char *>(Entry.Buffer.getBufferStart()),
           static_cast<int>(BufferSize), MPI_BYTE, Packed.data(), TotalSize,
           &Position, Comm);

  return Packed;
}

UnpackedMessage
MPISharedStorageCache::unpackMessage(const std::vector<char> &Buffer) {
  MPI_Comm Comm = CommHandle.get();
  int Position = 0;
  int TotalSize = static_cast<int>(Buffer.size());

  uint32_t HashSize = 0;
  MPI_Unpack(Buffer.data(), TotalSize, &Position, &HashSize, 1, MPI_UINT32_T,
             Comm);

  std::string HashStr(HashSize, '\0');
  MPI_Unpack(Buffer.data(), TotalSize, &Position, HashStr.data(),
             static_cast<int>(HashSize), MPI_CHAR, Comm);

  uint8_t IsDynLib = 0;
  MPI_Unpack(Buffer.data(), TotalSize, &Position, &IsDynLib, 1, MPI_BYTE, Comm);

  uint64_t DataSize = 0;
  MPI_Unpack(Buffer.data(), TotalSize, &Position, &DataSize, 1, MPI_UINT64_T,
             Comm);

  std::vector<char> Data(DataSize);
  MPI_Unpack(Buffer.data(), TotalSize, &Position, Data.data(),
             static_cast<int>(DataSize), MPI_BYTE, Comm);

  return UnpackedMessage{HashT(StringRef(HashStr)), std::move(Data),
                         IsDynLib != 0};
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
