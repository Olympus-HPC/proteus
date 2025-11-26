//===-- ObjectCacheChain.hpp -- Object cache chain header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_OBJECTCACHECHAIN_HPP
#define PROTEUS_OBJECTCACHECHAIN_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/MemoryBufferRef.h>

#include "proteus/Caching/ObjectCache.hpp"
#include "proteus/CompiledLibrary.hpp"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

class ObjectCacheChain {
public:
  explicit ObjectCacheChain(const std::string &Label);

  std::unique_ptr<CompiledLibrary> lookup(HashT &HashValue);
  void store(HashT &HashValue, const CacheEntry &Entry);
  void printStats();

private:
  void addCache(std::unique_ptr<ObjectCache> Cache);
  void buildFromConfig(const std::string &ConfigStr);
  std::unique_ptr<ObjectCache> createCache(const std::string &Name);

  std::vector<std::unique_ptr<ObjectCache>> Caches;
  const std::string Label;
  const std::string DistributedRank;
};

} // namespace proteus

#endif
