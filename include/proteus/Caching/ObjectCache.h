//===-- ObjectCache.h -- Object cache interface --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_OBJECTCACHE_H
#define PROTEUS_OBJECTCACHE_H

#include "proteus/CompiledLibrary.h"
#include "proteus/Hashing.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/MemoryBufferRef.h>

#include <cstdint>
#include <memory>
#include <string>
namespace proteus {

using namespace llvm;

struct CacheEntry {
  MemoryBufferRef Buffer;
  bool IsDynLib;

  static CacheEntry staticObject(MemoryBufferRef Buf) { return {Buf, false}; }

  static CacheEntry sharedObject(MemoryBufferRef Buf) { return {Buf, true}; }

  bool isSharedObject() const { return IsDynLib; }
  bool isStaticObject() const { return !IsDynLib; }
};

class ObjectCache {
public:
  virtual ~ObjectCache() = default;

  virtual std::string getName() const = 0;
  virtual std::unique_ptr<CompiledLibrary> lookup(const HashT &HashValue) = 0;
  virtual void store(const HashT &HashValue, const CacheEntry &Entry) = 0;
  virtual void printStats() = 0;
  virtual uint64_t getHits() const = 0;
  virtual uint64_t getAccesses() const = 0;

protected:
  ObjectCache() = default;
};

} // namespace proteus

#endif
