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
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include "llvm/ADT/StringRef.h"

#include "Utils.h"

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

  std::unique_ptr<MemoryBuffer> lookup(uint64_t HashValue,
                                       bool &UsesDeviceGlobals) {
    TIMESCOPE("object lookup");
    Accesses++;

    std::string Filebase =
        StorageDirectory + "/cache-jit-" + std::to_string(HashValue);

    auto JsonBuffer = MemoryBuffer::getFile(Filebase + ".json");
    if (!JsonBuffer)
      return nullptr;

    Expected<json::Value> CacheInfo =
        json::parse(JsonBuffer.get()->getBuffer());
    if (auto Err = CacheInfo.takeError())
      FATAL_ERROR("Cannot parse cache info from json file" + Filebase +
                  ".json");

    Hits++;

    UsesDeviceGlobals =
        CacheInfo->getAsObject()->getBoolean("UsesDeviceGlobals").value();
    if (UsesDeviceGlobals) {
      auto CacheBuf = MemoryBuffer::getFile(Filebase + ".bc");
      if (!CacheBuf)
        FATAL_ERROR("Expected bitcode cache file");

      return std::move(CacheBuf.get());
    }

    auto CacheBuf = MemoryBuffer::getFile(Filebase + ".o");
    if (!CacheBuf)
      FATAL_ERROR("Expected object cache file");

    return std::move(CacheBuf.get());
  }

  void store(uint64_t HashValue, bool UsesDeviceGlobals, StringRef Bitcode,
             MemoryBufferRef ObjBufRef) {
    TIMESCOPE("Store cache");

    std::string Filebase =
        StorageDirectory + "/cache-jit-" + std::to_string(HashValue);
    json::Object CacheInfo;

    CacheInfo["UsesDeviceGlobals"] = UsesDeviceGlobals;
    saveToFile(Filebase + ".json", json::Value(std::move(CacheInfo)));
    saveToFile(Filebase + ".bc", Bitcode);
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

  template <typename T> void saveToFile(StringRef Filepath, T &&Data) {
    std::error_code EC;
    raw_fd_ostream Out(Filepath, EC);
    if (EC)
      FATAL_ERROR("Cannot open file" + Filepath);
    Out << Data;
    Out.close();
  }
};

} // namespace proteus

#endif