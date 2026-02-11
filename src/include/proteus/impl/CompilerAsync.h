#ifndef PROTEUS_ASYNC_COMPILER_H
#define PROTEUS_ASYNC_COMPILER_H

#include "proteus/impl/CompilationTask.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Hashing.h"

#include <condition_variable>
#include <deque>
#include <thread>

namespace proteus {

using namespace llvm;

class CompilationResult {
public:
  explicit CompilationResult() : IsReadyFlag{false} {}

  CompilationResult(const CompilationResult &) = delete;
  CompilationResult &operator=(const CompilationResult &) = delete;

  CompilationResult(CompilationResult &&) noexcept = delete;
  CompilationResult &operator=(CompilationResult &&) noexcept = delete;

  bool isReady() { return IsReadyFlag; }

  void set(std::unique_ptr<MemoryBuffer> ObjBuf) {
    ResultObjBuf = std::move(ObjBuf);
    IsReadyFlag = true;
  }

  void wait() {
    // Busy wait until it's ready.
    while (!IsReadyFlag) {
      std::this_thread::yield();
    }
  }

  std::unique_ptr<MemoryBuffer> take() { return std::move(ResultObjBuf); }

private:
  std::atomic<bool> IsReadyFlag;
  std::unique_ptr<MemoryBuffer> ResultObjBuf;
};

class CompilerAsync {
public:
  void compile(CompilationTask &&CT) {
    std::unique_lock Lock{Mutex};
    // Get HashValue before the move of CT.
    HashT HashValue = CT.getHashValue();
    Worklist.emplace_back(std::move(CT));
    CompilationResultMap.emplace(HashValue,
                                 std::make_unique<CompilationResult>());
    CondVar.notify_one();
  }

  void run() {
    [[maybe_unused]] int Count = 0;
    while (Active) {
      std::unique_lock Lock(Mutex);
      CondVar.wait(Lock, [this] { return !Worklist.empty() || !Active; });
      if (!Active)
        break;
      if (Worklist.empty())
        reportFatalError("Expected non-empty Worklist");
      CompilationTask CT = std::move(Worklist.front());
      Worklist.pop_front();
      Lock.unlock();

      Count++;
      std::unique_ptr<MemoryBuffer> ObjBuf = CT.compile();

      // Ensure threads are joined before static objects are destroyed during
      // program exit. This static is initialized after COMGR's own statics
      // (triggered by compile() above), so C++ reverse destruction order
      // guarantees ~ShutdownGuard runs first.
      static ShutdownGuard Guard{this};

      Lock.lock();
      CompilationResultMap.at(CT.getHashValue())->set(std::move(ObjBuf));
      Lock.unlock();
    }

    PROTEUS_DBG(Logger::logs("proteus")
                << "Thread exiting! Compiled " + std::to_string(Count) + "\n");
  }

  void joinAllThreads() {
    {
      std::unique_lock Lock{Mutex};
      // Return if already inactive.
      if (!Active)
        return;

      Active = false;
    }
    CondVar.notify_all();

    for (auto &Thread : Threads)
      if (Thread.joinable())
        Thread.join();

    Threads.clear();
  }

  bool isCompilationPending(HashT HashValue) {
    std::unique_lock Lock{Mutex};
    return !(CompilationResultMap.find(HashValue) ==
             CompilationResultMap.end());
  }

  std::unique_ptr<MemoryBuffer> takeCompilationResult(HashT HashValue,
                                                      bool BlockingWait) {
    std::unique_lock Lock{Mutex};
    auto It = CompilationResultMap.find(HashValue);
    if (It == CompilationResultMap.end())
      return nullptr;

    std::unique_ptr<CompilationResult> &CRes = It->second;

    if (BlockingWait) {
      // Release the lock while waiting for the result.
      Lock.unlock();
      CRes->wait();
      // Reacquire the lock for synchronized access, the result is ready.
      Lock.lock();
    } else {
      if (!CRes->isReady())
        return nullptr;
    }

    // If compilation result is ready, take ownership of the buffer, erase it
    // from the compilation results map and move the buffer to the caller.
    std::unique_ptr<MemoryBuffer> ObjBuf = CRes->take();
    // Use the HashValue key as the iterator may have been invalidated by
    // insert/emplace from another thread.
    CompilationResultMap.erase(HashValue);
    Lock.unlock();
    return ObjBuf;
  }

  CompilerAsync(int NumThreads) {
    Active = true;
    for (int I = 0; I < NumThreads; ++I)
      Threads.emplace_back(&CompilerAsync::run, this);
  }

  ~CompilerAsync() { joinAllThreads(); }

private:
  // Joins threads before static objects are destroyed at exit (e.g.,
  // COMGR/HIPRTC). Must be constructed after the first compile()) so that
  // reverse destruction order ensures this runs first.
  struct ShutdownGuard {
    CompilerAsync *CA;
    ~ShutdownGuard() {
      if (CA)
        CA->joinAllThreads();
    }
  };

  bool Active;
  std::mutex Mutex;
  std::condition_variable CondVar;
  std::unordered_map<HashT, std::unique_ptr<CompilationResult>>
      CompilationResultMap;
  std::deque<CompilationTask> Worklist;
  std::vector<std::thread> Threads;
};

} // namespace proteus

#endif
