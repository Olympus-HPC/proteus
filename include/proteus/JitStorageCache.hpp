//===-- JitStorageCache.hpp -- JIT storage-based cache header impl. --===//
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
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include <llvm/ADT/StringRef.h>

#include "proteus/Hashing.hpp"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

// NOTE: Storage cache assumes that stored code is re-usable across runs!
// TODO: Source code changes should invalidate the cache. Also, if storing
// assembly (PTX) or binary (ELF), then device globals may have different
// addresses that render it invalid. In this case, store LLVM IR to re-link
// globals.
template <typename Function_t> class JitStorageCache {
public:
  JitStorageCache() { std::filesystem::create_directory(StorageDirectory); }

  std::unique_ptr<MemoryBuffer> lookup(HashT &HashValue) {
    TIMESCOPE("object lookup");
    Accesses++;

    std::string Filebase =
        StorageDirectory + "/cache-jit-" + HashValue.toString();

    auto CacheBuf = MemoryBuffer::getFileAsStream(Filebase + ".o");
    if (!CacheBuf)
      return nullptr;

    Hits++;
    return std::move(*CacheBuf);
  }

  void store(HashT &HashValue, MemoryBufferRef ObjBufRef) {
    TIMESCOPE("Store cache");

    std::string Filebase =
        StorageDirectory + "/cache-jit-" + HashValue.toString();

    saveToFile(Filebase + ".o", StringRef{ObjBufRef.getBufferStart(),
                                          ObjBufRef.getBufferSize()});
  }

  void printStats() {
    // Use printf to avoid re-ordering outputs by outs() in HIP.
    printf("JitStorageCache hits %lu total %lu\n", Hits, Accesses);
  }

private:
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory = ".proteus";
};

} // namespace proteus

#endif
