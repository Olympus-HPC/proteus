//===-- Utils.h -- Utilities header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_UTILS_H
#define PROTEUS_UTILS_H

#include "proteus/Error.h"
#include "proteus/Logger.h"
#include "proteus/TimeTracing.h"

#include <llvm/ADT/Twine.h>
#include <llvm/Support/SourceMgr.h>

#include <filesystem>
#include <string>

template <typename T>
inline void saveToFile(llvm::StringRef Filepath, T &&Data) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Filepath, EC);
  if (EC)
    proteus::reportFatalError("Cannot open file" + Filepath);
  Out << Data;
  Out.close();
}

// Write to temp file first, then rename so readers never see a partial file.
// Note: rename() atomicity is not guaranteed on NFS; concurrent readers may
// occasionally see missing files, causing redundant compilation.
template <typename T>
void saveToFileAtomic(llvm::StringRef Filepath, T &&Data) {
  std::string TempPath = Filepath.str() + ".tmp";

  std::error_code EC;
  llvm::raw_fd_ostream Out(TempPath, EC);
  if (EC)
    proteus::reportFatalError("Cannot open file " + TempPath);
  Out << Data;
  Out.close();

  std::filesystem::rename(TempPath, Filepath.str());
}

inline std::string getDistributedRank() {
  // Try commonly used environment variables to get the rank in distributed
  // runs.
  const char *Id = nullptr;

  // MPICH, Intel MPI, MVAPICH.
  if (!Id)
    Id = std::getenv("PMI_RANK");
  if (!Id)
    Id = std::getenv("MPI_RANK");

  // Open MPI.
  if (!Id)
    Id = std::getenv("OMPI_COMM_WORLD_RANK");

  // SLURM (if using srun).
  if (!Id)
    Id = std::getenv("SLURM_PROCID");

  // PBS/Torque.
  if (!Id)
    Id = std::getenv("PBS_TASKNUM");

  if (Id) {
    return std::string(Id);
  }

  // Fallback for non-distributed execution.
  return "0";
}

#if PROTEUS_ENABLE_HIP
#include "proteus/UtilsHIP.h"
#endif

#if PROTEUS_ENABLE_CUDA
#include "proteus/UtilsCUDA.h"
#endif

#endif
