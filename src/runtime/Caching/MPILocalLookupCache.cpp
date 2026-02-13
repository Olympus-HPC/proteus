//===-- MPILocalLookupCache.cpp -- MPI local-lookup cache impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/Caching/MPILocalLookupCache.h"

#include "proteus/impl/TimeTracing.h"

namespace proteus {

MPILocalLookupCache::MPILocalLookupCache(const std::string &Label)
    : MPIStorageCache(Label) {
  startCommThread();
}

std::unique_ptr<CompiledLibrary>
MPILocalLookupCache::lookup(const HashT &HashValue) {
  TIMESCOPE("MPILocalLookupCache::lookup");
  Accesses++;

  auto Result = lookupFromDisk(HashValue);
  if (Result)
    Hits++;

  return Result;
}

} // namespace proteus
