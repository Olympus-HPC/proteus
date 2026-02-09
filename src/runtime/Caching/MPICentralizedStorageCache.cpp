//===-- MPICentralizedStorageCache.cpp -- Centralized MPI cache impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/Caching/MPICentralizedStorageCache.h"

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

struct LookupRequest {
  HashT Hash;
};

struct LookupResponse {
  bool Found;
  bool IsDynLib;
  std::vector<char> Data;
};

//===----------------------------------------------------------------------===//
// MPICentralizedStorageCache implementation
//===----------------------------------------------------------------------===//

MPICentralizedStorageCache::MPICentralizedStorageCache(const std::string &Label)
    : StorageDirectory(Config::get().ProteusCacheDir
                           ? Config::get().ProteusCacheDir.value()
                           : ".proteus"),
      Label(Label) {
  std::filesystem::create_directories(StorageDirectory);
  startCommThread();
}

MPICentralizedStorageCache::~MPICentralizedStorageCache() { finalize(); }

void MPICentralizedStorageCache::finalize() {
  if (Finalized)
    return;

  int MPIFinalized = 0;
  MPI_Finalized(&MPIFinalized);
  if (MPIFinalized) {
    if (!PendingSends.empty()) {
      Logger::trace("[MPICentralizedStorageCache] Warning: MPI already "
                    "finalized, cannot complete " +
                    std::to_string(PendingSends.size()) + " pending sends\n");
    }
    CommThread.stop();
    Finalized = true;
    return;
  }

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPICentralizedStorageCache:" + Label + "] Rank " +
                  std::to_string(CommHandle.getRank()) +
                  " flushing, PendingSends=" +
                  std::to_string(PendingSends.size()) + "\n");
  }

  MPI_Comm Comm = CommHandle.get();

  completeAllPendingSends();

  MPI_Barrier(Comm);

  CommThread.stop();

  MPI_Barrier(Comm);

  Finalized = true;
}

std::unique_ptr<CompiledLibrary>
MPICentralizedStorageCache::lookup(const HashT &HashValue) {
  TIMESCOPE("MPICentralizedStorageCache::lookup");
  Accesses++;

  return lookupRemote(HashValue);
}

std::unique_ptr<CompiledLibrary>
MPICentralizedStorageCache::lookupRemote(const HashT &HashValue) {
  MPI_Comm Comm = CommHandle.get();

  auto ReqBuf = packLookupRequest(HashValue);
  MPI_Send(ReqBuf.data(), static_cast<int>(ReqBuf.size()), MPI_BYTE, 0,
           TagLookupRequest, Comm);

  MPI_Status Status;
  MPI_Probe(0, TagLookupResponse, Comm, &Status);

  int RespSize = 0;
  MPI_Get_count(&Status, MPI_BYTE, &RespSize);
  std::vector<char> RespBuf(RespSize);
  MPI_Recv(RespBuf.data(), RespSize, MPI_BYTE, 0, TagLookupResponse, Comm,
           MPI_STATUS_IGNORE);

  auto Resp = unpackLookupResponse(RespBuf);

  if (!Resp.Found)
    return nullptr;

  Hits++;

  auto MemBuf = MemoryBuffer::getMemBufferCopy(
      StringRef(Resp.Data.data(), Resp.Data.size()));
  if (Resp.IsDynLib) {
    std::string TempPath =
        StorageDirectory + "/cache-jit-" + HashValue.toString() + ".so";
    if (!std::filesystem::exists(TempPath)) {
      saveToFileAtomic(TempPath, MemBuf->getBuffer());
    }
    return std::make_unique<CompiledLibrary>(TempPath);
  }
  return std::make_unique<CompiledLibrary>(std::move(MemBuf));
}

std::unique_ptr<CompiledLibrary>
MPICentralizedStorageCache::lookupLocal(const HashT &HashValue) {
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

void MPICentralizedStorageCache::store(const HashT &HashValue,
                                       const CacheEntry &Entry) {
  TIMESCOPE("MPICentralizedStorageCache::store");

  forwardToWriter(HashValue, Entry);
}

void MPICentralizedStorageCache::communicationThreadMain() {
  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPICentralizedStorageCache:" + Label +
                  "] Communication thread started\n");
  }

  MPI_Comm Comm = CommHandle.get();

  while (true) {
    bool AnyActivity = false;
    int Flag = 0;
    MPI_Status Status;

    MPI_Iprobe(MPI_ANY_SOURCE, TagStore, Comm, &Flag, &Status);
    if (Flag) {
      AnyActivity = true;
      handleStoreMessage(Status);
    }

    MPI_Iprobe(MPI_ANY_SOURCE, TagLookupRequest, Comm, &Flag, &Status);
    if (Flag) {
      AnyActivity = true;
      handleLookupRequest(Status);
    }

    if (!AnyActivity) {
      if (CommThread.shutdownRequested()) {
        MPI_Iprobe(MPI_ANY_SOURCE, TagStore, Comm, &Flag, &Status);
        if (Flag) {
          handleStoreMessage(Status);
          continue;
        }
        MPI_Iprobe(MPI_ANY_SOURCE, TagLookupRequest, Comm, &Flag, &Status);
        if (Flag) {
          handleLookupRequest(Status);
          continue;
        }
        break;
      }
      CommThread.waitOrShutdown(std::chrono::milliseconds(1));
    }
  }

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPICentralizedStorageCache:" + Label +
                  "] Communication thread exiting\n");
  }
}

void MPICentralizedStorageCache::startCommThread() {
  if (CommHandle.getRank() != 0)
    return;
  CommThread.start([this] { communicationThreadMain(); });
}

void MPICentralizedStorageCache::handleStoreMessage(MPI_Status &Status) {
  MPI_Comm Comm = CommHandle.get();
  int MsgSize = 0;
  MPI_Get_count(&Status, MPI_BYTE, &MsgSize);

  std::vector<char> Buffer(MsgSize);
  MPI_Recv(Buffer.data(), MsgSize, MPI_BYTE, Status.MPI_SOURCE, TagStore, Comm,
           MPI_STATUS_IGNORE);

  auto Msg = unpackStoreMessage(Comm, Buffer);
  saveToDisk(Msg.Hash, Msg.Data.data(), Msg.Data.size(), Msg.IsDynLib);
}

void MPICentralizedStorageCache::handleLookupRequest(MPI_Status &Status) {
  MPI_Comm Comm = CommHandle.get();
  int SourceRank = Status.MPI_SOURCE;

  int MsgSize = 0;
  MPI_Get_count(&Status, MPI_BYTE, &MsgSize);
  std::vector<char> Buffer(MsgSize);
  MPI_Recv(Buffer.data(), MsgSize, MPI_BYTE, SourceRank, TagLookupRequest, Comm,
           MPI_STATUS_IGNORE);

  auto Req = unpackLookupRequest(Buffer);

  bool Found = false;
  bool IsDynLib = false;
  std::vector<char> Data;

  auto Result = lookupLocal(Req.Hash);
  if (Result) {
    IsDynLib = Result->IsDynLib;
    if (Result->ObjectModule) {
      auto BufRef = Result->ObjectModule->getMemBufferRef();
      Data.assign(BufRef.getBufferStart(),
                  BufRef.getBufferStart() + BufRef.getBufferSize());
      Found = true;
    } else {
      auto Buf = MemoryBuffer::getFileAsStream(Result->DynLibPath);
      if (Buf) {
        auto BufRef = (*Buf)->getMemBufferRef();
        Data.assign(BufRef.getBufferStart(),
                    BufRef.getBufferStart() + BufRef.getBufferSize());
        Found = true;
      }
    }
  }

  auto RespBuf = packLookupResponse(Found, IsDynLib, Data);
  MPI_Send(RespBuf.data(), static_cast<int>(RespBuf.size()), MPI_BYTE,
           SourceRank, TagLookupResponse, Comm);
}

std::vector<char>
MPICentralizedStorageCache::packLookupRequest(const HashT &HashValue) {
  MPI_Comm Comm = CommHandle.get();
  std::string HashStr = HashValue.toString();
  uint32_t HashSize = static_cast<uint32_t>(HashStr.size());

  int HashSizeBytes, HashStrBytes;
  MPI_Pack_size(1, MPI_UINT32_T, Comm, &HashSizeBytes);
  MPI_Pack_size(static_cast<int>(HashSize), MPI_CHAR, Comm, &HashStrBytes);

  int TotalSize = HashSizeBytes + HashStrBytes;
  std::vector<char> Packed(TotalSize);
  int Position = 0;

  MPI_Pack(&HashSize, 1, MPI_UINT32_T, Packed.data(), TotalSize, &Position,
           Comm);
  MPI_Pack(HashStr.data(), static_cast<int>(HashSize), MPI_CHAR, Packed.data(),
           TotalSize, &Position, Comm);

  return Packed;
}

std::vector<char>
MPICentralizedStorageCache::packLookupResponse(bool Found, bool IsDynLib,
                                               const std::vector<char> &Data) {
  MPI_Comm Comm = CommHandle.get();
  uint8_t FoundByte = Found ? 1 : 0;
  uint8_t IsDynLibByte = IsDynLib ? 1 : 0;
  uint64_t DataSize = Data.size();

  if (DataSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
    reportFatalError("Lookup response size exceeds MPI int limit: " +
                     std::to_string(DataSize) + " bytes");
  }

  int FoundBytes, IsDynLibBytes, DataSizeBytes, DataBytes;
  MPI_Pack_size(1, MPI_BYTE, Comm, &FoundBytes);
  MPI_Pack_size(1, MPI_BYTE, Comm, &IsDynLibBytes);
  MPI_Pack_size(1, MPI_UINT64_T, Comm, &DataSizeBytes);
  MPI_Pack_size(static_cast<int>(DataSize), MPI_BYTE, Comm, &DataBytes);

  int TotalSize = FoundBytes + IsDynLibBytes + DataSizeBytes + DataBytes;
  std::vector<char> Packed(TotalSize);
  int Position = 0;

  MPI_Pack(&FoundByte, 1, MPI_BYTE, Packed.data(), TotalSize, &Position, Comm);
  MPI_Pack(&IsDynLibByte, 1, MPI_BYTE, Packed.data(), TotalSize, &Position,
           Comm);
  MPI_Pack(&DataSize, 1, MPI_UINT64_T, Packed.data(), TotalSize, &Position,
           Comm);
  if (!Data.empty()) {
    MPI_Pack(const_cast<char *>(Data.data()), static_cast<int>(DataSize),
             MPI_BYTE, Packed.data(), TotalSize, &Position, Comm);
  }

  return Packed;
}

LookupRequest MPICentralizedStorageCache::unpackLookupRequest(
    const std::vector<char> &Buffer) {
  MPI_Comm Comm = CommHandle.get();
  int Position = 0;
  int TotalSize = static_cast<int>(Buffer.size());

  uint32_t HashSize = 0;
  MPI_Unpack(Buffer.data(), TotalSize, &Position, &HashSize, 1, MPI_UINT32_T,
             Comm);

  std::string HashStr(HashSize, '\0');
  MPI_Unpack(Buffer.data(), TotalSize, &Position, HashStr.data(),
             static_cast<int>(HashSize), MPI_CHAR, Comm);

  return LookupRequest{HashT(StringRef(HashStr))};
}

LookupResponse MPICentralizedStorageCache::unpackLookupResponse(
    const std::vector<char> &Buffer) {
  MPI_Comm Comm = CommHandle.get();
  int Position = 0;
  int TotalSize = static_cast<int>(Buffer.size());

  uint8_t FoundByte = 0;
  MPI_Unpack(Buffer.data(), TotalSize, &Position, &FoundByte, 1, MPI_BYTE,
             Comm);

  uint8_t IsDynLibByte = 0;
  MPI_Unpack(Buffer.data(), TotalSize, &Position, &IsDynLibByte, 1, MPI_BYTE,
             Comm);

  uint64_t DataSize = 0;
  MPI_Unpack(Buffer.data(), TotalSize, &Position, &DataSize, 1, MPI_UINT64_T,
             Comm);

  if (DataSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
    reportFatalError("Lookup response size exceeds MPI int limit: " +
                     std::to_string(DataSize) + " bytes");
  }

  std::vector<char> Data(DataSize);
  if (DataSize > 0) {
    MPI_Unpack(Buffer.data(), TotalSize, &Position, Data.data(),
               static_cast<int>(DataSize), MPI_BYTE, Comm);
  }

  return LookupResponse{FoundByte != 0, IsDynLibByte != 0, std::move(Data)};
}

void MPICentralizedStorageCache::saveToDisk(const HashT &HashValue,
                                            const char *Data, size_t Size,
                                            bool IsDynLib) {
  std::string Filebase =
      StorageDirectory + "/cache-jit-" + HashValue.toString();
  std::string Extension = IsDynLib ? ".so" : ".o";
  std::string Filepath = Filebase + Extension;

  if (std::filesystem::exists(Filepath))
    return;

  saveToFile(Filepath, StringRef{Data, Size});

  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPICentralizedStorageCache] Saved " + Filepath + " (" +
                  std::to_string(Size) + " bytes)\n");
  }
}

void MPICentralizedStorageCache::forwardToWriter(const HashT &HashValue,
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
                      /*dest=*/0, TagStore, Comm, &Pending->Request);
  if (Err != MPI_SUCCESS) {
    reportFatalError("MPI_Isend failed with error code " + std::to_string(Err));
  }

  PendingSends.push_back(std::move(Pending));
  pollPendingSends();
}

void MPICentralizedStorageCache::pollPendingSends() {
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

void MPICentralizedStorageCache::completeAllPendingSends() {
  for (auto &Pending : PendingSends) {
    MPI_Wait(&Pending->Request, MPI_STATUS_IGNORE);
  }
  PendingSends.clear();
}

void MPICentralizedStorageCache::printStats() {
  printf(
      "[proteus][%s] MPICentralizedStorageCache rank %d/%d hits %lu accesses "
      "%lu\n",
      Label.c_str(), CommHandle.getRank(), CommHandle.getSize(), Hits,
      Accesses);
}

} // namespace proteus
