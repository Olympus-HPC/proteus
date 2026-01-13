//===-- StorageCache.cpp -- Storage cache implementation --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/Caching/StorageCache.hpp"
#include "proteus/CompiledLibrary.hpp"
#include "proteus/Config.hpp"
#include "proteus/Hashing.hpp"
#include "proteus/Utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include <cstdint>
#include <filesystem>

namespace proteus {

using namespace llvm;

// NOTE: Storage cache assumes that stored code is re-usable across runs!
// TODO: Source code changes should invalidate the cache. Also, if storing
// assembly (PTX) or binary (ELF), then device globals may have different
// addresses that render it invalid. In this case, store LLVM IR to re-link
// globals.
StorageCache::StorageCache(const std::string &Label)
    : StorageDirectory(Config::get().ProteusCacheDir
                           ? Config::get().ProteusCacheDir.value()
                           : ".proteus"),
      Label(Label), DistributedRank(getDistributedRank()) {
  std::filesystem::create_directory(StorageDirectory);
}

std::unique_ptr<CompiledLibrary> StorageCache::lookup(const HashT &HashValue) {
  TIMESCOPE("object lookup");
  Accesses++;

  std::string Filebase = StorageDirectory + "/" + DistributedRank +
                         "-cache-jit-" + HashValue.toString();

  // We first try to load a relocatable object file to create the code
  // library. If that fails, we try to find a dynamic library file to setup
  // the code library. If both fail, this hash is not cached.
  auto CacheBuf = MemoryBuffer::getFileAsStream(Filebase + ".o");
  if (CacheBuf) {
    Hits++;
    return std::make_unique<CompiledLibrary>(std::move(*CacheBuf));
  }

  if (std::filesystem::exists(Filebase + ".so")) {
    Hits++;
    return std::make_unique<CompiledLibrary>(Filebase + ".so");
  }

  return nullptr;
}

void StorageCache::store(const HashT &HashValue, const CacheEntry &Entry) {
  TIMESCOPE("Store cache");

  std::string Filebase = StorageDirectory + "/" + DistributedRank +
                         "-cache-jit-" + HashValue.toString();
  std::string Extension = Entry.isSharedObject() ? ".so" : ".o";

  saveToFile(Filebase + Extension, StringRef{Entry.Buffer.getBufferStart(),
                                             Entry.Buffer.getBufferSize()});
}

void StorageCache::printStats() {
  printf("[proteus][%s] StorageCache rank %s hits %lu accesses %lu\n",
         Label.c_str(), DistributedRank.c_str(), Hits, Accesses);
}

} // namespace proteus
