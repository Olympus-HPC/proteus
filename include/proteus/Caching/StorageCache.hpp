//===-- StorageCache.hpp -- Storage cache header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITSTOREDCACHE_HPP
#define PROTEUS_JITSTOREDCACHE_HPP

#include <cstdint>
#include <filesystem>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include <llvm/ADT/StringRef.h>

#include "proteus/CompiledLibrary.hpp"
#include "proteus/Config.hpp"
#include "proteus/Hashing.hpp"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

// NOTE: Storage cache assumes that stored code is re-usable across runs!
// TODO: Source code changes should invalidate the cache. Also, if storing
// assembly (PTX) or binary (ELF), then device globals may have different
// addresses that render it invalid. In this case, store LLVM IR to re-link
// globals.
class StorageCache {
public:
  StorageCache(const std::string &Label);

  std::unique_ptr<CompiledLibrary> lookup(HashT &HashValue);

  void store(HashT &HashValue, MemoryBufferRef ObjBufRef);

  void storeDynamicLibrary(HashT &HashValue, const SmallString<128> &Path);

  void printStats();

private:
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory;
  const std::string Label;
  const std::string DistributedRank;
};

} // namespace proteus

#endif
