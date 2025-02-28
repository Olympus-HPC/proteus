#ifndef PROTEUS_ASYNC_COMPILER_HPP
#define PROTEUS_ASYNC_COMPILER_HPP

#include <condition_variable>
#include <thread>

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include "proteus/CompilationTask.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

class CompilerAsync {
public:
  static CompilerAsync &instance(int NumThreads) {
    static CompilerAsync Singleton{NumThreads};
    return Singleton;
  }

  void compile(CompilationTask &&CT) {
    std::unique_lock Lock{Mutex};
    Worklist.emplace_back(std::move(CT));
    CompilationResults.emplace(CT.getHashValue(), nullptr);
    CondVar.notify_one();
  }

  void run() {
#if PROTEUS_ENABLE_CUDA
    // CUDA requires a thread context.
    CUdevice CUDev;
    CUcontext CUCtx;

    CUresult CURes = cuCtxGetDevice(&CUDev);
    if (CURes == CUDA_ERROR_INVALID_CONTEXT or !CUDev)
      // TODO: is selecting device 0 correct?
      proteusCuErrCheck(cuDeviceGet(&CUDev, 0));

    proteusCuErrCheck(cuCtxGetCurrent(&CUCtx));
    if (!CUCtx)
      proteusCuErrCheck(cuCtxCreate(&CUCtx, 0, CUDev));
#endif

    int Count = 0;
    while (Active) {
      std::unique_lock Lock(Mutex);
      CondVar.wait(Lock, [&]() { return !Worklist.empty() || !Active; });
      if (!Active)
        break;
      CompilationTask CT = std::move(Worklist.back());
      Worklist.pop_back();
      Lock.unlock();

      Count++;
      std::unique_ptr<MemoryBuffer> ObjBuf = CT.compile();
      Lock.lock();
      CompilationResults.at(CT.getHashValue()) = std::move(ObjBuf);
      Lock.unlock();
    }

    PROTEUS_DBG(Logger::logs("proteus")
                << "Thread exiting! Compiled " + std::to_string(Count) + "\n");
  }

  void joinAllThreads() {
    Active = false;
    CondVar.notify_all();

    for (auto &Thread : Threads)
      Thread.join();

    Threads.clear();
  }

  bool isCompilationPending(HashT HashValue) {
    return !(CompilationResults.find(HashValue) == CompilationResults.end());
  }

  std::unique_ptr<MemoryBuffer> getCompilationResult(HashT HashValue) {
    auto It = CompilationResults.find(HashValue);
    if (It == CompilationResults.end())
      return nullptr;

    std::unique_lock Lock{Mutex};
    std::unique_ptr<MemoryBuffer> &ObjBuf = CompilationResults.at(HashValue);
    if (!ObjBuf)
      return nullptr;

    return std::move(ObjBuf);
  }

private:
  std::atomic<bool> Active;
  std::mutex Mutex;
  std::condition_variable CondVar;
  std::unordered_map<HashT, std::unique_ptr<MemoryBuffer>> CompilationResults;
  std::vector<CompilationTask> Worklist;
  std::vector<std::thread> Threads;

  CompilerAsync(int NumThreads) {
    Active = true;
    for (int I = 0; I < NumThreads; ++I)
      Threads.emplace_back(&CompilerAsync::run, this);
  }

  ~CompilerAsync() {
    if (Threads.size() > 0)
      joinAllThreads();
  }
};

} // namespace proteus

#endif