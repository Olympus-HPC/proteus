//===-- ObjectCache.hpp -- Object cache interface --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_OBJECTCACHE_HPP
#define PROTEUS_OBJECTCACHE_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/MemoryBufferRef.h>

#include "proteus/CompiledLibrary.hpp"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

struct CacheEntry {
  MemoryBufferRef Buffer;
  bool IsDynLib;

  static CacheEntry objectFile(MemoryBufferRef Buf) { return {Buf, false}; }

  static CacheEntry sharedObject(MemoryBufferRef Buf) { return {Buf, true}; }

  bool isDynLib() const { return IsDynLib; }
  bool isObject() const { return !IsDynLib; }
};

class ObjectCache {
public:
  virtual ~ObjectCache() = default;

  virtual std::string getName() const = 0;
  virtual std::unique_ptr<CompiledLibrary> lookup(HashT &HashValue) = 0;
  virtual void store(HashT &HashValue, const CacheEntry &Entry) = 0;
  virtual void printStats() = 0;
  virtual uint64_t getHits() const = 0;
  virtual uint64_t getAccesses() const = 0;

protected:
  ObjectCache() = default;
};

} // namespace proteus

#endif
