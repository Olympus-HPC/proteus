//===-- MPILocalLookupCache.h -- MPI local-lookup cache header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MPI cache where stores are forwarded to rank 0 but lookups read from the
// local shared filesystem. Suitable when all ranks share a filesystem.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_MPILOCALLOOKUP_CACHE_H
#define PROTEUS_MPILOCALLOOKUP_CACHE_H

#include "proteus/impl/Caching/MPIStorageCache.h"

namespace proteus {

class MPILocalLookupCache : public MPIStorageCache {
public:
  MPILocalLookupCache(const std::string &Label);

  std::string getName() const override { return "MPILocalLookup"; }

  std::unique_ptr<CompiledLibrary> lookup(const HashT &HashValue) override;
};

} // namespace proteus

#endif
