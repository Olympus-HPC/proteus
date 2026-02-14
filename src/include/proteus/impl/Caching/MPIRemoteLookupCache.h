//===-- MPIRemoteLookupCache.h -- MPI remote-lookup cache header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A fully centralized MPI storage cache where rank 0 is the single writer AND
// reader. Other ranks make synchronous MPI requests to rank 0 for both store()
// and lookup() operations.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_MPIREMOTELOOKUP_CACHE_H
#define PROTEUS_MPIREMOTELOOKUP_CACHE_H

#include "proteus/impl/Caching/MPIStorageCache.h"

#include <vector>

namespace proteus {

struct LookupRequest {
  HashT Hash;
};

struct LookupResponse {
  bool Found;
  bool IsDynLib;
  std::vector<char> Data;
};

class MPIRemoteLookupCache : public MPIStorageCache {
public:
  MPIRemoteLookupCache(const std::string &Label);

  std::string getName() const override { return "MPIRemoteLookup"; }

  std::unique_ptr<CompiledLibrary> lookup(const HashT &HashValue) override;

protected:
  void handleMessage(MPI_Status &Status, MPITag Tag) override;

private:
  std::unique_ptr<CompiledLibrary> lookupRemote(const HashT &HashValue);

  void handleLookupRequest(MPI_Status &Status);

  std::vector<char> packLookupRequest(const HashT &HashValue);
  std::vector<char> packLookupResponse(bool Found, bool IsDynLib,
                                       const std::vector<char> &Data);
  LookupRequest unpackLookupRequest(const std::vector<char> &Buffer);
  LookupResponse unpackLookupResponse(const std::vector<char> &Buffer);
};

} // namespace proteus

#endif
