//===-- CacheAdapter.hpp -- Adapter for cache migration --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Adapters to help migrate from JitCache and JitStorageCache to the new Cache.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_CACHE_ADAPTER_HPP
#define PROTEUS_CACHE_ADAPTER_HPP

#include <memory>
#include <string>

#include "proteus/Cache.hpp"
#include "proteus/CompilationResult.hpp"
#include "proteus/Hashing.hpp"

namespace proteus {

/**
 * @brief Adapter to make a legacy JitCache use the new Cache interface
 * 
 * This adapter is used to migrate from JitCache to the new Cache architecture.
 * It wraps a modern Cache implementation and exposes it via the old JitCache interface.
 */
template <typename Function_t>
class JitCacheAdapter {
public:
  /**
   * @brief Construct a JitCacheAdapter
   * 
   * @param Config Cache configuration
   */
  explicit JitCacheAdapter(const CacheConfig& Config = CacheConfig())
      : TheCache(Cache::create(Config)) {}

  /**
   * @brief Look up a function pointer by hash
   * 
   * @param HashValue Hash to look up
   * @return Function pointer if found, nullptr otherwise
   */
  Function_t lookup(HashT& HashValue) {
    auto Result = TheCache->lookup(HashValue);
    if (!Result) {
      return nullptr;
    }
    
    Hits++;
    Accesses++;
    return static_cast<Function_t>(Result->getFunctionPtr());
  }

  /**
   * @brief Insert a function into the cache
   * 
   * @param HashValue Hash of the function
   * @param FunctionPtr Function pointer
   * @param FnName Function name
   * @param RCArr Runtime constants
   */
  void insert(HashT& HashValue, Function_t FunctionPtr, StringRef FnName,
              ArrayRef<RuntimeConstant> RCArr) {
    // Create a dummy compilation result with just the function pointer
    std::unique_ptr<CompilationResult> Result = std::make_unique<CompilationResult>(
        HashValue,
        FnName.str(),
        nullptr, // No object buffer needed
        static_cast<void*>(FunctionPtr),
        SmallVector<RuntimeConstant>(RCArr.begin(), RCArr.end()));
    
    TheCache->store(std::move(Result));
    Accesses++;
  }

  /**
   * @brief Print cache statistics
   */
  void printStats() {
    TheCache->printStats();
  }

private:
  std::unique_ptr<Cache> TheCache;
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
};

/**
 * @brief Adapter to make a legacy JitStorageCache use the new Cache interface
 * 
 * This adapter is used to migrate from JitStorageCache to the new Cache architecture.
 * It wraps a modern DiskCache implementation and exposes it via the old JitStorageCache interface.
 */
template <typename Function_t>
class JitStorageCacheAdapter {
public:
  /**
   * @brief Construct a JitStorageCacheAdapter
   * 
   * @param CachePath Directory to store cache files
   */
  explicit JitStorageCacheAdapter(const std::string& CachePath = ".proteus") {
    CacheConfig Config;
    Config.UseMemoryCache = false;
    Config.UseDiskCache = true;
    Config.DiskCachePath = CachePath;
    TheCache = std::unique_ptr<DiskCache>(new DiskCache(CachePath));
  }

  /**
   * @brief Look up a memory buffer by hash
   * 
   * @param HashValue Hash to look up
   * @return Memory buffer if found, nullptr otherwise
   */
  std::unique_ptr<MemoryBuffer> lookup(HashT& HashValue) {
    auto Result = TheCache->lookup(HashValue);
    if (!Result) {
      return nullptr;
    }
    
    Hits++;
    Accesses++;
    return Result->takeObjectBuffer();
  }

  /**
   * @brief Store a memory buffer in the cache
   * 
   * @param HashValue Hash of the function
   * @param ObjBufRef Memory buffer reference
   */
  void store(HashT& HashValue, MemoryBufferRef ObjBufRef) {
    // Create a memory buffer copy
    std::unique_ptr<MemoryBuffer> ObjBuf = 
        MemoryBuffer::getMemBufferCopy(ObjBufRef.getBuffer(), ObjBufRef.getBufferIdentifier());
    
    // Create a minimal compilation result with just the object buffer
    std::unique_ptr<CompilationResult> Result = std::make_unique<CompilationResult>(
        HashValue,
        "unknown", // Name not needed for disk cache
        std::move(ObjBuf),
        nullptr, // Function pointer not needed for disk cache
        SmallVector<RuntimeConstant>{}); // Runtime constants not needed for disk cache
    
    TheCache->store(std::move(Result));
    Accesses++;
  }

  /**
   * @brief Print cache statistics
   */
  void printStats() {
    TheCache->printStats();
  }

private:
  std::unique_ptr<DiskCache> TheCache;
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
};

} // namespace proteus

#endif // PROTEUS_CACHE_ADAPTER_HPP