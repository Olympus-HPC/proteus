#ifndef PROTEUS_JIT_DEV_HPP
#define PROTEUS_JIT_DEV_HPP

#include <llvm/ADT/StringRef.h>
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
  std::unique_ptr<MemoryBuffer> CompiledObject;

  std::deque<std::unique_ptr<FuncBase>> Functions;
  TargetModelType TargetModel;
  std::string TargetTriple;
  Dispatcher &Dispatch;

  bool IsCompiled = false;

  template <typename... ArgT> struct KernelHandle {
    Func<void, ArgT...> &F;
    JitModule &M;

    // Launch with type-safety.
    [[nodiscard]] auto launch(LaunchDims Grid, LaunchDims Block,
                              uint64_t ShmemBytes, void *Stream, ArgT... Args) {
      // Create the type-safe tuple.
      auto Tup = std::make_tuple(static_cast<ArgT>(Args)...);

      // Create the ArrayRef<void*> pointing at each tuple element.
      std::array<void *, sizeof...(ArgT)> Ptrs;
      std::apply(
          [&](auto &...Elts) {
            size_t I = 0;
            ((Ptrs[I++] = (void *)&Elts), ...);
          },
          Tup);

      // Call launch through module.
      // TODO: should it use the dispatcher directly?
      return M.launch(F, Grid, Block, Ptrs, ShmemBytes, Stream);
    }

    FuncBase *operator->() { return &F; }
  };

  TargetModelType parseTargetModel(StringRef Target) {
    if (Target == "host" || Target == "native") {
      return TargetModelType::HOST;
    }

    if (Target == "cuda") {
      return TargetModelType::CUDA;
    }

    if (Target == "hip") {
      return TargetModelType::HIP;
    }

    PROTEUS_FATAL_ERROR("Unsupported target " + Target);
  }

  std::string getTargetTriple(TargetModelType Model) {
    switch (Model) {
    case TargetModelType::HOST:
      return sys::getProcessTriple();
    case TargetModelType::CUDA:
      return "nvptx64-nvidia-cuda";
    case TargetModelType::HIP:
      return "amdgcn-amd-amdhsa";
    default:
      PROTEUS_FATAL_ERROR("Unsupported target model");
    }
  }

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

  DispatchResult launch(FuncBase &F, LaunchDims GridDim, LaunchDims BlockDim,
                        ArrayRef<void *> KernelArgs, uint64_t ShmemSize,
                        void *Stream) {
    if (!IsCompiled)
      compile();

    return Dispatch.launch(F.getName(), GridDim, BlockDim, KernelArgs,
                           ShmemSize, Stream);
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
    if (Verify)
      if (verifyModule(*Mod, &errs())) {
        PROTEUS_FATAL_ERROR("Broken module found, JIT compilation aborted!");
      }

    Dispatch.compile(std::move(Ctx), std::move(Mod));
    IsCompiled = true;
  }

  void print() { Mod->print(outs(), nullptr); }
};

template <typename RetT, typename... ArgT> void FuncBase::call(StringRef Name) {
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

  return Dispatch.run<RetT>(getName(), Args...);
}

} // namespace proteus

#endif
