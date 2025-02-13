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

#include <string>

#include <llvm/ADT/Twine.h>
#include <llvm/Support/SourceMgr.h>

#include "proteus/Error.h"
#include "proteus/Logger.hpp"
#include "proteus/TimeTracing.hpp"

template <typename T> void saveToFile(llvm::StringRef Filepath, T &&Data) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Filepath, EC);
  if (EC)
    FATAL_ERROR("Cannot open file" + Filepath);
  Out << Data;
  Out.close();
}

#if PROTEUS_ENABLE_HIP
#include "proteus/UtilsHIP.h"
#endif

#if PROTEUS_ENABLE_CUDA
#include "proteus/UtilsCUDA.h"
#endif

#endif
