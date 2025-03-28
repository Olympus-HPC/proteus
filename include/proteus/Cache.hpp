//===-- Cache.hpp -- JIT compilation cache header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Caches for JIT compiled functions.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_CACHE_HPP
#define PROTEUS_CACHE_HPP

#include <filesystem>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "proteus/CompilationResult.hpp"
#include "proteus/Hashing.hpp"
#include "proteus/TimeTracing.hpp"
#include "proteus/Utils.h"

namespace proteus {

/**
 * @brief Cache configuration options
 */
struct CacheConfig {
  bool UseMemoryCache = true;
  bool UseDiskCache = false;
  std::string DiskCachePath = ".proteus";
  bool EnableStats = true;
};

/**
 * @brief Abstract base class for caching CompilationResults
 */
class Cache {
public:
  /**
   * @brief Create an appropriate cache based on configuration
   */
  static std::unique_ptr<Cache> create(const CacheConfig& Config);

  /**
   * @brief Virtual destructor
   */
  virtual ~Cache() = default;

  /**
   * @brief Look up a compilation result by hash
   * 
   * @param HashValue Hash to look up
   * @return Compilation result if found, nullptr otherwise
   */
  virtual std::unique_ptr<CompilationResult> lookup(const HashT& HashValue) = 0;

  /**
   * @brief Store a compilation result
   * 
   * @param Result Compilation result to store
   */
  virtual void store(std::unique_ptr<CompilationResult> Result) = 0;

  /**
   * @brief Print cache statistics
   */
  virtual void printStats() const = 0;

protected:
  /**
   * @brief Cache statistics
   */
  struct CacheStats {
    uint64_t Hits = 0;
    uint64_t Accesses = 0;

    /**
     * @brief Record a cache access
     * 
     * @param Hit Whether the access was a hit
     */
    void recordAccess(bool Hit) {
      Accesses++;
      if (Hit) {
        Hits++;
      }
    }

    /**
     * @brief Get hit rate as a percentage
     */
    double getHitRate() const {
      return Accesses > 0 ? (static_cast<double>(Hits) / Accesses) * 100.0 : 0.0;
    }
  };

  CacheStats Stats;
  bool EnableStats;

  Cache(bool EnableStats) : EnableStats(EnableStats) {}
};

/**
 * @brief In-memory cache for CompilationResults
 */
class MemoryCache : public Cache {
public:
  /**
   * @brief Construct a memory cache
   * 
   * @param EnableStats Whether to collect statistics
   */
  explicit MemoryCache(bool EnableStats = true) : Cache(EnableStats) {}

  /**
   * @brief Look up a compilation result by hash
   * 
   * @param HashValue Hash to look up
   * @return Compilation result if found, nullptr otherwise
   */
  std::unique_ptr<CompilationResult> lookup(const HashT& HashValue) override {
    TIMESCOPE("MemoryCache::lookup");

    std::lock_guard<std::mutex> Lock(Mutex);
    if (EnableStats) {
      Stats.Accesses++;
    }
    
    auto It = Entries.find(HashValue);
    if (It == Entries.end()) {
      return nullptr;
    }

    if (EnableStats) {
      Stats.Hits++;
      It->second.NumHits++;
    }
    
    // Return a copy, keep original in cache
    auto& Entry = It->second;
    return std::make_unique<CompilationResult>(
        HashValue,
        Entry.Result->getMangledName(),
        nullptr, // Object buffer is not needed for in-memory cache
        Entry.Result->getFunctionPtr(),
        Entry.Result->getRuntimeConstants());
  }

  /**
   * @brief Store a compilation result
   * 
   * @param Result Compilation result to store
   */
  void store(std::unique_ptr<CompilationResult> Result) override {
    if (!Result) {
      return;
    }

    std::lock_guard<std::mutex> Lock(Mutex);
    HashT HashValue = Result->getHashValue();
    
    // Store entry in cache
    auto& Entry = Entries[HashValue];
    Entry.Result = std::move(Result);
    Entry.NumExecs = 1;
    Entry.NumHits = 0;
  }

  /**
   * @brief Print cache statistics
   */
  void printStats() const override {
    if (!EnableStats) {
      return;
    }

    std::lock_guard<std::mutex> Lock(Mutex);
    printf("MemoryCache hits %lu total %lu (%.2f%%)\n", 
           Stats.Hits, Stats.Accesses, Stats.getHitRate());
    
    for (const auto& [HashValue, Entry] : Entries) {
      printf("HashValue %s NumExecs %lu NumHits %lu\n",
             HashValue.toString().c_str(), Entry.NumExecs, Entry.NumHits);
    }
  }

private:
  /**
   * @brief In-memory cache entry
   */
  struct CacheEntry {
    std::unique_ptr<CompilationResult> Result;
    uint64_t NumExecs = 0;
    uint64_t NumHits = 0;
  };

  std::unordered_map<HashT, CacheEntry> Entries;
  mutable std::mutex Mutex;
};

/**
 * @brief Persistent disk cache for CompilationResults
 */
class DiskCache : public Cache {
public:
  /**
   * @brief Construct a disk cache
   * 
   * @param CachePath Directory to store cache files
   * @param EnableStats Whether to collect statistics
   */
  explicit DiskCache(std::string CachePath = ".proteus", bool EnableStats = true)
      : Cache(EnableStats), CachePath(std::move(CachePath)) {
    std::filesystem::create_directory(this->CachePath);
  }

  /**
   * @brief Look up a compilation result by hash
   * 
   * @param HashValue Hash to look up
   * @return Compilation result if found, nullptr otherwise
   */
  std::unique_ptr<CompilationResult> lookup(const HashT& HashValue) override {
    TIMESCOPE("DiskCache::lookup");
    
    if (EnableStats) {
      std::lock_guard<std::mutex> Lock(Mutex);
      Stats.Accesses++;
    }

    std::string Filebase = getCacheFilePath(HashValue);

    auto CacheBuf = MemoryBuffer::getFile(Filebase + ".o");
    if (!CacheBuf) {
      return nullptr;
    }

    // In a real implementation, we'd deserialize metadata and get the function pointer
    // For now, we're just returning a minimal result with the object buffer
    
    if (EnableStats) {
      std::lock_guard<std::mutex> Lock(Mutex);
      Stats.Hits++;
    }
    
    return std::make_unique<CompilationResult>(
        HashValue,
        "unknown_mangled_name",  // Would come from metadata
        std::move(CacheBuf.get()),
        nullptr,  // Function pointer would come from loading the object
        SmallVector<RuntimeConstant>{}); // Would come from metadata
  }

  /**
   * @brief Store a compilation result
   * 
   * @param Result Compilation result to store
   */
  void store(std::unique_ptr<CompilationResult> Result) override {
    if (!Result || !Result->getObjectBuffer().getBufferSize()) {
      return;
    }

    TIMESCOPE("DiskCache::store");

    HashT HashValue = Result->getHashValue();
    std::string Filebase = getCacheFilePath(HashValue);

    // Store object file
    saveToFile(Filebase + ".o", 
              StringRef{Result->getObjectBuffer().getBufferStart(),
                       Result->getObjectBuffer().getBufferSize()});
    
    // In a real implementation, we'd also store metadata including:
    // - Mangled name
    // - Runtime constants
    // - Any other data needed to recreate the function pointer
  }

  /**
   * @brief Print cache statistics
   */
  void printStats() const override {
    if (!EnableStats) {
      return;
    }

    std::lock_guard<std::mutex> Lock(Mutex);
    printf("DiskCache hits %lu total %lu (%.2f%%)\n", 
           Stats.Hits, Stats.Accesses, Stats.getHitRate());
  }

private:
  /**
   * @brief Get the cache file path for a hash
   */
  std::string getCacheFilePath(const HashT& HashValue) const {
    return CachePath + "/cache-jit-" + HashValue.toString();
  }

  std::string CachePath;
  mutable std::mutex Mutex;
};

/**
 * @brief Hierarchical cache combining memory and disk caches
 */
class HierarchicalCache : public Cache {
public:
  /**
   * @brief Construct a hierarchical cache
   * 
   * @param Config Cache configuration
   */
  explicit HierarchicalCache(const CacheConfig& Config)
      : Cache(Config.EnableStats) {
    if (Config.UseMemoryCache) {
      MemCache = std::unique_ptr<MemoryCache>(new MemoryCache(Config.EnableStats));
    }
    if (Config.UseDiskCache) {
      DiskCache = std::unique_ptr<class DiskCache>(new class DiskCache(Config.DiskCachePath, Config.EnableStats));
    }
  }

  /**
   * @brief Look up a compilation result by hash
   * 
   * @param HashValue Hash to look up
   * @return Compilation result if found, nullptr otherwise
   */
  std::unique_ptr<CompilationResult> lookup(const HashT& HashValue) override {
    TIMESCOPE("HierarchicalCache::lookup");
    
    if (EnableStats) {
      Stats.Accesses++;
    }

    // First check memory cache
    if (MemCache) {
      auto Result = MemCache->lookup(HashValue);
      if (Result) {
        if (EnableStats) {
          Stats.Hits++;
        }
        return Result;
      }
    }

    // Then check disk cache
    if (DiskCache) {
      auto Result = DiskCache->lookup(HashValue);
      if (Result) {
        // Add to memory cache for next time
        if (MemCache) {
          MemCache->store(std::make_unique<CompilationResult>(
              Result->getHashValue(),
              Result->getMangledName(),
              nullptr, // Object buffer not needed for memory cache
              Result->getFunctionPtr(),
              Result->getRuntimeConstants()));
        }
        
        if (EnableStats) {
          Stats.Hits++;
        }
        return Result;
      }
    }

    return nullptr;
  }

  /**
   * @brief Store a compilation result
   * 
   * @param Result Compilation result to store
   */
  void store(std::unique_ptr<CompilationResult> Result) override {
    if (!Result) {
      return;
    }

    // Store in memory cache
    if (MemCache) {
      MemCache->store(std::make_unique<CompilationResult>(
          Result->getHashValue(),
          Result->getMangledName(),
          nullptr, // Object buffer not needed for memory cache
          Result->getFunctionPtr(),
          Result->getRuntimeConstants()));
    }

    // Store in disk cache
    if (DiskCache) {
      DiskCache->store(std::move(Result));
    }
  }

  /**
   * @brief Print cache statistics
   */
  void printStats() const override {
    if (!EnableStats) {
      return;
    }

    printf("HierarchicalCache hits %lu total %lu (%.2f%%)\n", 
           Stats.Hits, Stats.Accesses, Stats.getHitRate());
    
    if (MemCache) {
      MemCache->printStats();
    }
    
    if (DiskCache) {
      DiskCache->printStats();
    }
  }

private:
  std::unique_ptr<MemoryCache> MemCache;
  std::unique_ptr<class DiskCache> DiskCache;
};

inline std::unique_ptr<Cache> Cache::create(const CacheConfig& Config) {
  if (Config.UseMemoryCache && Config.UseDiskCache) {
    return std::unique_ptr<Cache>(new HierarchicalCache(Config));
  } else if (Config.UseMemoryCache) {
    return std::unique_ptr<Cache>(new MemoryCache(Config.EnableStats));
  } else if (Config.UseDiskCache) {
    return std::unique_ptr<Cache>(new DiskCache(Config.DiskCachePath, Config.EnableStats));
  } else {
    // Default to memory cache if none specified
    return std::unique_ptr<Cache>(new MemoryCache(Config.EnableStats));
  }
}

} // namespace proteus

#endif // PROTEUS_CACHE_HPP