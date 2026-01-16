//===-- StorageCache.h -- Storage cache header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITSTOREDCACHE_H
#define PROTEUS_JITSTOREDCACHE_H

#include "proteus/Caching/ObjectCache.h"
#include "proteus/CompiledLibrary.h"
#include "proteus/Hashing.h"

#include <llvm/Support/MemoryBufferRef.h>

#include <cstdint>
namespace proteus {

using namespace llvm;

// NOTE: Storage cache assumes that stored code is re-usable across runs!
// TODO: Source code changes should invalidate the cache. Also, if storing
// assembly (PTX) or binary (ELF), then device globals may have different
// addresses that render it invalid. In this case, store LLVM IR to re-link
// globals.
class StorageCache : public ObjectCache {
public:
  StorageCache(const std::string &Label);

  std::string getName() const override { return "Storage"; }

  std::unique_ptr<CompiledLibrary> lookup(const HashT &HashValue) override;

  void store(const HashT &HashValue, const CacheEntry &Entry) override;

  void printStats() override;

  uint64_t getHits() const override { return Hits; }

  uint64_t getAccesses() const override { return Accesses; }

private:
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory;
  const std::string Label;
  const std::string DistributedRank;
};

} // namespace proteus

#endif
