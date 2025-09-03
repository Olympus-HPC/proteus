#ifndef PROTEUS_JIT_DEV_HPP
#define PROTEUS_JIT_DEV_HPP

#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <deque>

#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/Frontend/Func.hpp"
#include "proteus/Frontend/TypeMap.hpp"

#include <iostream>

namespace proteus {
using namespace llvm;

class JitModule {
private:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> Mod;
  std::unique_ptr<CompiledLibrary> Library;

  std::deque<std::unique_ptr<FuncBase>> Functions;
  TargetModelType TargetModel;
  std::string TargetTriple;
  Dispatcher &Dispatch;

  HashT ModuleHash = 0;
  bool IsCompiled = false;

  template <typename... ArgT> struct KernelHandle {
    Func<void, ArgT...> &F;
    JitModule &M;

    // Launch with type-safety.
    [[nodiscard]] auto launch(LaunchDims Grid, LaunchDims Block,
                              uint64_t ShmemBytes, void *Stream, ArgT... Args) {
      // Pointers to the local parameter copies.
      void *Ptrs[sizeof...(ArgT)] = {(void *)&Args...};

      if (!M.isCompiled())
        M.compile();

      auto GetKernelFunc = [&]() {
        // Get the kernel func pointer directly from the Func object if
        // available.
        if (auto KernelFunc = F.getCompiledFunc()) {
          return KernelFunc;
        }

        // Get the kernel func pointer from the Dispatch and store it to the
        // Func object to avoid cache lookups.
        // TODO: Re-think caching and dispatchers.
        auto KernelFunc = reinterpret_cast<decltype(F.getCompiledFunc())>(
            M.Dispatch.getFunctionAddress(F.getName(), M.ModuleHash,
                                          M.getLibrary()));

        F.setCompiledFunc(KernelFunc);

        return KernelFunc;
      };

      return M.Dispatch.launch(reinterpret_cast<void *>(GetKernelFunc()), Grid,
                               Block, Ptrs, ShmemBytes, Stream);
    }

    FuncBase *operator->() { return &F; }
  };

  bool isDeviceModule() {
    return ((TargetModel == TargetModelType::CUDA) ||
            (TargetModel == TargetModelType::HIP));
  }

  void setKernel(FuncBase &F) {
    switch (TargetModel) {
    case TargetModelType::CUDA: {
      NamedMDNode *MD = Mod->getOrInsertNamedMetadata("nvvm.annotations");

      Metadata *MDVals[] = {
          ConstantAsMetadata::get(F.getFunction()),
          MDString::get(*Ctx, "kernel"),
          ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(*Ctx), 1))};
      // Append metadata to nvvm.annotations.
      MD->addOperand(MDNode::get(*Ctx, MDVals));

      // Add a function attribute for the kernel.
      F.getFunction()->addFnAttr(Attribute::get(*Ctx, "kernel"));
      return;
    }
    case TargetModelType::HIP:
      F.getFunction()->setCallingConv(CallingConv::AMDGPU_KERNEL);
      return;
    case TargetModelType::HOST:
      PROTEUS_FATAL_ERROR("Host does not support setKernel");
    default:
      PROTEUS_FATAL_ERROR("Unsupported target " + TargetTriple);
    }
  }

public:
  JitModule(StringRef Target = "host")
      : Ctx{std::make_unique<LLVMContext>()},
        Mod{std::make_unique<Module>("JitModule", *Ctx)},
        TargetModel{parseTargetModel(Target)},
        TargetTriple(getTargetTriple(TargetModel)),
        Dispatch(Dispatcher::getDispatcher(TargetModel)) {}

  // Disable copy and move constructors.
  JitModule(const JitModule &) = delete;
  JitModule &operator=(const JitModule &) = delete;
  JitModule(JitModule &&) = delete;
  JitModule &operator=(JitModule &&) = delete;

  template <typename RetT, typename... ArgT>
  Func<RetT, ArgT...> &addFunction(StringRef Name) {
    if (IsCompiled)
      PROTEUS_FATAL_ERROR(
          "The module is compiled, no further code can be added");

    Mod->setTargetTriple(TargetTriple);
    FunctionCallee FC;
    FC = Mod->getOrInsertFunction(Name, TypeMap<RetT>::get(*Ctx),
                                  TypeMap<ArgT>::get(*Ctx)...);
    Function *F = dyn_cast<Function>(FC.getCallee());
    if (!F)
      PROTEUS_FATAL_ERROR("Unexpected");
    auto TypedFn = std::make_unique<Func<RetT, ArgT...>>(*this, FC, Dispatch);
    Func<RetT, ArgT...> &TypedFnRef = *TypedFn;
    std::unique_ptr<FuncBase> &Fn = Functions.emplace_back(std::move(TypedFn));

    Fn->declArgs<ArgT...>();
    return TypedFnRef;
  }

  bool isCompiled() const { return IsCompiled; }

  const Module &getModule() const { return *Mod; }

  template <typename... ArgT> KernelHandle<ArgT...> addKernel(StringRef Name) {
    if (IsCompiled)
      PROTEUS_FATAL_ERROR(
          "The module is compiled, no further code can be added");

    if (!isDeviceModule())
      PROTEUS_FATAL_ERROR("Expected a device module for addKernel");

    Mod->setTargetTriple(TargetTriple);
    FunctionCallee FC;
    FC = Mod->getOrInsertFunction(Name, TypeMap<void>::get(*Ctx),
                                  TypeMap<ArgT>::get(*Ctx)...);
    Function *F = dyn_cast<Function>(FC.getCallee());
    if (!F)
      PROTEUS_FATAL_ERROR("Unexpected");
    auto TypedFn = std::make_unique<Func<void, ArgT...>>(*this, FC, Dispatch);
    Func<void, ArgT...> &TypedFnRef = *TypedFn;
    std::unique_ptr<FuncBase> &Fn = Functions.emplace_back(std::move(TypedFn));

    Fn->declArgs<ArgT...>();

    setKernel(*Fn);
    return KernelHandle<ArgT...>{TypedFnRef, *this};
  }

  void compile(bool Verify = false) {
    if (IsCompiled)
      return;

    if (Verify)
      if (verifyModule(*Mod, &errs())) {
        PROTEUS_FATAL_ERROR("Broken module found, JIT compilation aborted!");
      }

    SmallVector<char, 0> Buffer;
    raw_svector_ostream OS(Buffer);
    WriteBitcodeToFile(*Mod, OS);

    // Create a unique module hash based on the bitcode and append to all
    // function names to make them unique.
    // TODO: Is this necessary?
    ModuleHash = hash(StringRef{Buffer.data(), Buffer.size()});
    for (auto &JitF : Functions) {
      JitF->setName(JitF->getName().str() + "$" + ModuleHash.toString());
    }

    if ((Library = Dispatch.lookupCompiledLibrary(ModuleHash))) {
      IsCompiled = true;
      return;
    }

    Library = std::make_unique<CompiledLibrary>(
        Dispatch.compile(std::move(Ctx), std::move(Mod), ModuleHash));
    IsCompiled = true;
  }

  HashT getModuleHash() const { return ModuleHash; }

  Dispatcher &getDispatcher() const { return Dispatch; }

  TargetModelType getTargetModel() const { return TargetModel; }

  CompiledLibrary &getLibrary() {
    if (!IsCompiled)
      compile();

    if (!Library)
      PROTEUS_FATAL_ERROR("Expected non-null library after compilation");

    return *Library;
  }

  void print() { Mod->print(outs(), nullptr); }
};

template <typename RetT, typename... ArgT>
std::enable_if_t<!std::is_void_v<RetT>, Var &> FuncBase::call(StringRef Name) {
  auto *F = getFunction();
  Module &M = *F->getParent();
  LLVMContext &Ctx = F->getContext();
  FunctionCallee Callee = M.getOrInsertFunction(Name, TypeMap<RetT>::get(Ctx),
                                                TypeMap<ArgT>::get(Ctx)...);
  Var &Ret = declVarInternal("ret", TypeMap<RetT>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

template <typename RetT, typename... ArgT>
std::enable_if_t<std::is_void_v<RetT>, void> FuncBase::call(StringRef Name) {
  auto *F = getFunction();
  Module &M = *F->getParent();
  LLVMContext &Ctx = F->getContext();
  FunctionCallee Callee = M.getOrInsertFunction(Name, TypeMap<RetT>::get(Ctx),
                                                TypeMap<ArgT>::get(Ctx)...);
  IRB.CreateCall(Callee);
}

template <typename RetT, typename... ArgT>
RetT Func<RetT, ArgT...>::operator()(ArgT... Args) {
  if (!J.isCompiled())
    J.compile();

  if (!CompiledFunc) {
    CompiledFunc = reinterpret_cast<decltype(CompiledFunc)>(
        J.getDispatcher().getFunctionAddress(getName(), J.getModuleHash(),
                                             J.getLibrary()));
  }

  if (J.getTargetModel() != TargetModelType::HOST)
    PROTEUS_FATAL_ERROR(
        "Target is a GPU model, cannot directly run functions, use launch()");

  if constexpr (std::is_void_v<RetT>)
    Dispatch.run<RetT(ArgT...)>(reinterpret_cast<void *>(CompiledFunc),
                                Args...);
  else
    return Dispatch.run<RetT(ArgT...)>(reinterpret_cast<void *>(CompiledFunc),
                                       Args...);
}

} // namespace proteus

#endif
