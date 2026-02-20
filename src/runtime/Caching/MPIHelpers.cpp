//===-- MPIHelpers.cpp -- MPI helper classes for cache implementations --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/Caching/MPIHelpers.h"

#include "proteus/Error.h"
#include "proteus/impl/Caching/ObjectCache.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Logger.h"

#include <llvm/ADT/StringRef.h>

#include <mpi.h>

#include <limits>

namespace proteus {

//===----------------------------------------------------------------------===//
// MPI Validation
//===----------------------------------------------------------------------===//

void validateMPIConfig() {
  int MPIInitialized = 0;
  proteusMpiCheck(MPI_Initialized(&MPIInitialized));
  if (!MPIInitialized) {
    reportFatalError("MPI caching requires MPI to be initialized. Call "
                     "MPI_Init_thread() before any JIT compilation.");
  }
  int Provided = 0;
  proteusMpiCheck(MPI_Query_thread(&Provided));
  if (Provided != MPI_THREAD_MULTIPLE) {
    reportFatalError("MPI caching requires MPI_THREAD_MULTIPLE "
                     "(provided level: " +
                     std::to_string(Provided) +
                     "). Initialize MPI with MPI_Init_thread()");
  }
}

//===----------------------------------------------------------------------===//
// MPICommHandle implementation
//===----------------------------------------------------------------------===//

MPICommHandle::MPICommHandle() {
  validateMPIConfig();

  proteusMpiCheck(MPI_Comm_dup(MPI_COMM_WORLD, &Comm));
  proteusMpiCheck(MPI_Comm_rank(Comm, &Rank));
  proteusMpiCheck(MPI_Comm_size(Comm, &Size));

  if (Config::get().traceSpecializations()) {
    Logger::trace("[MPICommHandle] Initialized communicator for rank " +
                  std::to_string(Rank) + "/" + std::to_string(Size) + "\n");
  }
}

MPICommHandle::~MPICommHandle() { free(); }

MPI_Comm MPICommHandle::get() const { return Comm; }

int MPICommHandle::getRank() const { return Rank; }

int MPICommHandle::getSize() const { return Size; }

void MPICommHandle::free() {
  if (Comm == MPI_COMM_NULL)
    return;

  int MPIFinalized = 0;
  proteusMpiCheck(MPI_Finalized(&MPIFinalized));
  if (MPIFinalized)
    reportFatalError(
        "[MPICommHandle] MPI finalized before communicator cleanup.");

  proteusMpiCheck(MPI_Comm_free(&Comm));
}

CommThreadHandle::~CommThreadHandle() { join(); }

void CommThreadHandle::join() {
  if (!Thread)
    return;

  if (Thread->joinable())
    Thread->join();
  Thread.reset();
  Running = false;
}

bool CommThreadHandle::isRunning() const { return Running; }

std::vector<char> packStoreMessage(MPI_Comm Comm, const HashT &HashValue,
                                   const CacheEntry &Entry) {
  std::string HashStr = HashValue.toString();
  uint32_t HashSize = static_cast<uint32_t>(HashStr.size());
  uint8_t IsDynLib = Entry.isSharedObject() ? 1 : 0;
  uint64_t BufferSize = Entry.Buffer.getBufferSize();

  if (BufferSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
    reportFatalError("Buffer size exceeds MPI int limit: " +
                     std::to_string(BufferSize) + " bytes");
  }

  int HashSizeBytes, HashStrBytes, FlagBytes, BufSizeBytes, DataBytes;
  proteusMpiCheck(MPI_Pack_size(1, MPI_UINT32_T, Comm, &HashSizeBytes));
  proteusMpiCheck(
      MPI_Pack_size(static_cast<int>(HashSize), MPI_CHAR, Comm, &HashStrBytes));
  proteusMpiCheck(MPI_Pack_size(1, MPI_BYTE, Comm, &FlagBytes));
  proteusMpiCheck(MPI_Pack_size(1, MPI_UINT64_T, Comm, &BufSizeBytes));
  proteusMpiCheck(
      MPI_Pack_size(static_cast<int>(BufferSize), MPI_BYTE, Comm, &DataBytes));

  int TotalSize =
      HashSizeBytes + HashStrBytes + FlagBytes + BufSizeBytes + DataBytes;
  std::vector<char> Packed(TotalSize);
  int Position = 0;

  proteusMpiCheck(MPI_Pack(&HashSize, 1, MPI_UINT32_T, Packed.data(), TotalSize,
                           &Position, Comm));

  proteusMpiCheck(MPI_Pack(HashStr.data(), static_cast<int>(HashSize), MPI_CHAR,
                           Packed.data(), TotalSize, &Position, Comm));

  proteusMpiCheck(MPI_Pack(&IsDynLib, 1, MPI_BYTE, Packed.data(), TotalSize,
                           &Position, Comm));

  proteusMpiCheck(MPI_Pack(&BufferSize, 1, MPI_UINT64_T, Packed.data(),
                           TotalSize, &Position, Comm));

  proteusMpiCheck(MPI_Pack(Entry.Buffer.getBufferStart(),
                           static_cast<int>(BufferSize), MPI_BYTE,
                           Packed.data(), TotalSize, &Position, Comm));

  return Packed;
}

StoreMessage unpackStoreMessage(MPI_Comm Comm,
                                const std::vector<char> &Buffer) {
  int Position = 0;
  int TotalSize = static_cast<int>(Buffer.size());

  uint32_t HashSize = 0;
  proteusMpiCheck(MPI_Unpack(Buffer.data(), TotalSize, &Position, &HashSize, 1,
                             MPI_UINT32_T, Comm));

  std::string HashStr(HashSize, '\0');
  proteusMpiCheck(MPI_Unpack(Buffer.data(), TotalSize, &Position,
                             HashStr.data(), static_cast<int>(HashSize),
                             MPI_CHAR, Comm));

  uint8_t IsDynLib = 0;
  proteusMpiCheck(MPI_Unpack(Buffer.data(), TotalSize, &Position, &IsDynLib, 1,
                             MPI_BYTE, Comm));

  uint64_t DataSize = 0;
  proteusMpiCheck(MPI_Unpack(Buffer.data(), TotalSize, &Position, &DataSize, 1,
                             MPI_UINT64_T, Comm));

  std::vector<char> Data(DataSize);
  proteusMpiCheck(MPI_Unpack(Buffer.data(), TotalSize, &Position, Data.data(),
                             static_cast<int>(DataSize), MPI_BYTE, Comm));

  return StoreMessage{HashT(llvm::StringRef(HashStr)), std::move(Data),
                      IsDynLib != 0};
}

} // namespace proteus
