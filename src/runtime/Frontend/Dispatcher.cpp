#include "proteus/Frontend/Dispatcher.h"
#include "proteus/Error.h"
#include "proteus/impl/Caching/ObjectCacheChain.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Frontend/DispatcherHost.h"
#include "proteus/impl/Hashing.h"
#if PROTEUS_ENABLE_HIP
#include "proteus/impl/Frontend/DispatcherHIP.h"
#include "proteus/impl/Frontend/DispatcherHostHIP.h"
#endif
#if PROTEUS_ENABLE_CUDA
#include "proteus/impl/Frontend/DispatcherCUDA.h"
#include "proteus/impl/Frontend/DispatcherHostCUDA.h"
#endif

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

namespace {

Dispatcher &getHostHIPDispatcher() {
#if PROTEUS_ENABLE_HIP
  return DispatcherHostHIP::instance();
#else
  reportFatalError("HIP support is not enabled");
#endif
}

Dispatcher &getHostCUDADispatcher() {
#if PROTEUS_ENABLE_CUDA
  return DispatcherHostCUDA::instance();
#else
  reportFatalError("CUDA support is not enabled");
#endif
}

Dispatcher &getHostDispatcher() { return DispatcherHost::instance(); }

Dispatcher &getHIPDispatcher() {
#if PROTEUS_ENABLE_HIP
  return DispatcherHIP::instance();
#else
  reportFatalError("HIP support is not enabled");
#endif
}

Dispatcher &getCUDADispatcher() {
#if PROTEUS_ENABLE_CUDA
  return DispatcherCUDA::instance();
#else
  reportFatalError("CUDA support is not enabled");
#endif
}
} // anonymous namespace

Dispatcher &Dispatcher::getDispatcher(TargetModelType TargetModel) {
  switch (TargetModel) {
  case TargetModelType::HOST_HIP:
    return getHostHIPDispatcher();
  case TargetModelType::HOST_CUDA:
    return getHostCUDADispatcher();
  case TargetModelType::HOST:
    return getHostDispatcher();
  case TargetModelType::HIP:
    return getHIPDispatcher();
  case TargetModelType::CUDA:
    return getCUDADispatcher();
  default:
    reportFatalError("Unsupported model type");
  }
}

Dispatcher::Dispatcher(const std::string &Name, TargetModelType TM)
    : TargetModel(TM), Label(Name) {
  if (Config::get().ProteusUseStoredCache)
    ObjectCache = std::make_unique<ObjectCacheChain>(Name);
}

Dispatcher::~Dispatcher() = default;

void Dispatcher::printObjectCacheStats() {
  if (Config::get().traceCacheStats() && ObjectCache)
    ObjectCache->printStats();
}

std::unique_ptr<llvm::MemoryBuffer>
Dispatcher::compile(std::unique_ptr<llvm::LLVMContext> Ctx,
                    std::unique_ptr<llvm::Module> M, const HashT &ModuleHash,
                    const CompileOptions &Opts) {
  // Keep the context alive for as long as the module. Setting [[maybe_unused]]
  // can trigger a lifetime bug.
  auto CtxOwner = std::move(Ctx);
  auto ModOwner = std::move(M);

  std::unique_ptr<llvm::MemoryBuffer> ObjectModule =
      compileModule(*ModOwner, Opts);
  if (!ObjectModule)
    reportFatalError("Expected non-null object library");

  registerObject(ModuleHash, ObjectModule->getMemBufferRef());

  return ObjectModule;
}

std::unique_ptr<CompiledLibrary>
Dispatcher::lookupCompiledLibrary(const HashT &ModuleHash) {
  if (!ObjectCache)
    return nullptr;
  return ObjectCache->lookup(ModuleHash);
}

KernelName::KernelName(const StringRef &Base) : Base(Base.str()) {}

KernelName::KernelName(std::string Base, const HashT &Specialization)
    : Base(std::move(Base)), Specialization(Specialization.toMangledSuffix()) {}

void *Dispatcher::getOrInsertFunction(const KernelName &Name,
                                      const HashT &ModuleHash,
                                      CompiledLibrary &Library) {
  if (void *FuncPtr = lookupFunction(Name, ModuleHash))
    return FuncPtr;
  return insertFunction(Name, ModuleHash, Library);
}

void Dispatcher::registerObject(const HashT &HashValue,
                                const llvm::MemoryBufferRef &Obj) {
  if (!ObjectCache)
    return;
  ObjectCache->store(HashValue, CacheEntry::staticObject(Obj));
}

} // namespace proteus
