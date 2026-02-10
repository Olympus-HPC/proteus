//===-- MPIRemoteLookupCache.cpp -- MPI remote-lookup cache impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/Caching/MPIRemoteLookupCache.h"

#include "proteus/Error.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/TimeTracing.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>

#include <mpi.h>

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

MPIRemoteLookupCache::MPIRemoteLookupCache(const std::string &Label)
    : MPIStorageCache(Label, /*StoreTag=*/0) {}

std::unique_ptr<CompiledLibrary>
MPIRemoteLookupCache::lookup(const HashT &HashValue) {
  TIMESCOPE("MPIRemoteLookupCache::lookup");
  Accesses++;

  return lookupRemote(HashValue);
}

std::unique_ptr<CompiledLibrary>
MPIRemoteLookupCache::lookupRemote(const HashT &HashValue) {
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

void MPIRemoteLookupCache::communicationThreadMain() {
  if (Config::get().ProteusTraceOutput >= 1) {
    Logger::trace("[MPIRemoteLookup:" + Label +
                  "] Communication thread started\n");
  }

  MPI_Comm Comm = CommHandle.get();

  while (true) {
    bool AnyActivity = false;
    int Flag = 0;
    MPI_Status Status;

    MPI_Iprobe(MPI_ANY_SOURCE, StoreTag, Comm, &Flag, &Status);
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
        MPI_Iprobe(MPI_ANY_SOURCE, StoreTag, Comm, &Flag, &Status);
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
    Logger::trace("[MPIRemoteLookup:" + Label +
                  "] Communication thread exiting\n");
  }
}

void MPIRemoteLookupCache::handleStoreMessage(MPI_Status &Status) {
  MPI_Comm Comm = CommHandle.get();
  int MsgSize = 0;
  MPI_Get_count(&Status, MPI_BYTE, &MsgSize);

  std::vector<char> Buffer(MsgSize);
  MPI_Recv(Buffer.data(), MsgSize, MPI_BYTE, Status.MPI_SOURCE, StoreTag, Comm,
           MPI_STATUS_IGNORE);

  auto Msg = unpackStoreMessage(Comm, Buffer);
  saveToDisk(Msg.Hash, Msg.Data.data(), Msg.Data.size(), Msg.IsDynLib);
}

void MPIRemoteLookupCache::handleLookupRequest(MPI_Status &Status) {
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

  auto Result = lookupFromDisk(Req.Hash);
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
MPIRemoteLookupCache::packLookupRequest(const HashT &HashValue) {
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
MPIRemoteLookupCache::packLookupResponse(bool Found, bool IsDynLib,
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
    MPI_Pack(Data.data(), static_cast<int>(DataSize), MPI_BYTE, Packed.data(),
             TotalSize, &Position, Comm);
  }

  return Packed;
}

LookupRequest
MPIRemoteLookupCache::unpackLookupRequest(const std::vector<char> &Buffer) {
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

LookupResponse
MPIRemoteLookupCache::unpackLookupResponse(const std::vector<char> &Buffer) {
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

} // namespace proteus
