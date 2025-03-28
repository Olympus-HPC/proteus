//===-- AsyncBuilder.hpp -- Asynchronous Builder header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AsyncBuilder implements the Builder interface for asynchronous compilation
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_ASYNC_BUILDER_HPP
#define PROTEUS_ASYNC_BUILDER_HPP

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "proteus/Builder.hpp"
#include "proteus/CompilationTask.hpp"
#include "proteus/CompilationResult.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"
#include "proteus/Logger.hpp"

namespace proteus {

/**
 * @brief Internal class for tracking asynchronous compilation results
 */
class AsyncCompilationResult {
public:
  explicit AsyncCompilationResult() : IsReady(false) {}

  AsyncCompilationResult(const AsyncCompilationResult&) = delete;
  AsyncCompilationResult& operator=(const AsyncCompilationResult&) = delete;
  AsyncCompilationResult(AsyncCompilationResult&&) = delete;
  AsyncCompilationResult& operator=(AsyncCompilationResult&&) = delete;

  /**
   * @brief Check if the result is ready
   */
  bool isReady() const { return IsReady; }

  /**
   * @brief Set the result
   */
  void set(std::unique_ptr<CompilationResult> Result) {
    this->Result = std::move(Result);
    IsReady = true;
  }

  /**
   * @brief Wait until the result is ready
   */
  void wait() const {
    // Busy wait until it's ready
    while (!IsReady) {
      std::this_thread::yield();
    }
  }

  /**
   * @brief Take ownership of the result
   */
  std::unique_ptr<CompilationResult> take() {
    return std::move(Result);
  }

private:
  std::atomic<bool> IsReady;
  std::unique_ptr<CompilationResult> Result;
};

/**
 * @brief Asynchronous Builder implementation
 * 
 * The AsyncBuilder executes compilation tasks asynchronously in a thread pool.
 */
class AsyncBuilder : public Builder {
public:
  /**
   * @brief Get the singleton instance of AsyncBuilder
   * 
   * @param NumThreads Number of worker threads to use
   */
  static AsyncBuilder& instance(int NumThreads = 4) {
    static AsyncBuilder Singleton(NumThreads);
    return Singleton;
  }

  /**
   * @brief Destructor
   */
  ~AsyncBuilder() override {
    if (!Threads.empty()) {
      joinAllThreads();
    }
  }

  /**
   * @brief Build a CompilationResult from a CompilationTask
   * 
   * Queues the task for asynchronous compilation and returns immediately.
   * The caller must check if the result is ready and wait if necessary.
   * 
   * @param Task The task containing all information needed for compilation
   * @return nullptr (async builds don't return results directly)
   */
  std::unique_ptr<CompilationResult> build(const CompilationTask& Task) override {
    // Asynchronous builds don't return results directly
    // Instead, we queue the task and return nullptr
    std::unique_lock<std::mutex> Lock(Mutex);
    Worklist.push_back(Task);
    CompilationResultMap.emplace(Task.getHashValue(),
                             std::make_unique<AsyncCompilationResult>());
    CondVar.notify_one();
    return nullptr;
  }

  /**
   * @brief Check if a compilation is pending
   * 
   * @param HashValue Hash of the compilation task
   * @return True if the compilation is pending
   */
  bool isCompilationPending(const HashT& HashValue) {
    std::unique_lock<std::mutex> Lock(Mutex);
    return CompilationResultMap.find(HashValue) != CompilationResultMap.end();
  }

  /**
   * @brief Get a compilation result, optionally waiting for it to complete
   * 
   * @param HashValue Hash of the compilation task
   * @param BlockingWait Whether to wait for the result to be ready
   * @return The compilation result, or nullptr if not ready and not waiting
   */
  std::unique_ptr<CompilationResult> getResult(const HashT& HashValue, bool BlockingWait = false) {
    std::unique_lock<std::mutex> Lock(Mutex);
    auto It = CompilationResultMap.find(HashValue);
    if (It == CompilationResultMap.end()) {
      return nullptr;
    }

    std::unique_ptr<AsyncCompilationResult>& Result = It->second;
    Lock.unlock();

    if (BlockingWait) {
      Result->wait();
    } else if (!Result->isReady()) {
      return nullptr;
    }

    // Result is ready, take ownership and remove from map
    std::unique_ptr<CompilationResult> RetVal = Result->take();
    Lock.lock();
    CompilationResultMap.erase(HashValue);
    return RetVal;
  }

  /**
   * @brief Join all worker threads
   */
  void joinAllThreads() {
    Active = false;
    CondVar.notify_all();

    for (auto& Thread : Threads) {
      Thread.join();
    }

    Threads.clear();
  }

private:
  std::atomic<bool> Active;
  std::mutex Mutex;
  std::condition_variable CondVar;
  std::unordered_map<HashT, std::unique_ptr<AsyncCompilationResult>> CompilationResultMap;
  std::deque<CompilationTask> Worklist;
  std::vector<std::thread> Threads;

  /**
   * @brief Construct an AsyncBuilder
   * 
   * @param NumThreads Number of worker threads to use
   */
  explicit AsyncBuilder(int NumThreads) : Active(true) {
    for (int I = 0; I < NumThreads; ++I) {
      Threads.emplace_back(&AsyncBuilder::workerThread, this);
    }
  }

  /**
   * @brief Worker thread function
   */
  void workerThread() {
    int Count = 0;
    while (Active) {
      std::unique_lock<std::mutex> Lock(Mutex);
      CondVar.wait(Lock, [this] { return !Worklist.empty() || !Active; });
      
      if (!Active) {
        break;
      }
      
      if (Worklist.empty()) {
        PROTEUS_FATAL_ERROR("Expected non-empty Worklist");
      }
      
      CompilationTask Task = Worklist.front();
      Worklist.pop_front();
      Lock.unlock();

      Count++;
      
      // Compile the task
      std::unique_ptr<MemoryBuffer> ObjBuffer = Task.compile();
      
      // Create a CompilationResult
      std::unique_ptr<CompilationResult> Result = std::make_unique<CompilationResult>(
          Task.getHashValue(),
          Task.getKernelName() + Task.getSuffix(),
          std::move(ObjBuffer),
          nullptr, // Function pointer would come from the compiled object
          Task.getRCValues());
      
      // Set the result
      Lock.lock();
      auto It = CompilationResultMap.find(Task.getHashValue());
      if (It != CompilationResultMap.end()) {
        It->second->set(std::move(Result));
      }
      Lock.unlock();
    }

    PROTEUS_DBG(Logger::logs("proteus")
                << "Thread exiting! Compiled " + std::to_string(Count) + "\n");
  }
};

} // namespace proteus

#endif // PROTEUS_ASYNC_BUILDER_HPP