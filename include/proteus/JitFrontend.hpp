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
  std::unique_ptr<MemoryBuffer> ObjectModule;

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
            M.Dispatch.getFunctionAddress(F.getName(), M.getObjectModuleRef()));

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
    // TODO: This is not needed for GPU JIT modules since they are separate
    // objects. However, CPU JIT modules end up in the same object through the
    // ORC JIT singleton. Reconsider the CPU JIT process.
    ModuleHash = hash(StringRef{Buffer.data(), Buffer.size()});
    for (auto &JitF : Functions) {
      JitF->setName(JitF->getName().str() + "$" + ModuleHash.toString());
    }

    if ((ObjectModule = Dispatch.lookupObjectModule(ModuleHash))) {
      IsCompiled = true;
      return;
    }

    ObjectModule = Dispatch.compile(std::move(Ctx), std::move(Mod), ModuleHash);
    IsCompiled = true;
  }

  HashT getModuleHash() const { return ModuleHash; }

  std::optional<MemoryBufferRef> getObjectModuleRef() const {
    // For host JIT modules the ObjectModule is alway nullptr and unused by
    // DispatcherHOST since it is unused by ORC JIT.
    if (!ObjectModule)
      return std::nullopt;

    return ObjectModule->getMemBufferRef();
  }

  const Dispatcher &getDispatcher() const { return Dispatch; }

  TargetModelType getTargetModel() const { return TargetModel; }

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

template <typename ThenLambda>
void FuncBase::If(Var &CondVar, ThenLambda &&Then, const char *File, int Line) {
  Function *F = getFunction();
  BasicBlock *CurBlock = IP.getBlock();
  BasicBlock *NextBlock =
      CurBlock->splitBasicBlock(IP.getPoint(), CurBlock->getName() + ".split");

  auto ContIP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  Scopes.emplace_back(File, Line, ScopeKind::IF, ContIP);

  BasicBlock *ThenBlock =
      BasicBlock::Create(F->getContext(), "if.then", F, NextBlock);
  BasicBlock *ExitBlock =
      BasicBlock::Create(F->getContext(), "if.cont", F, NextBlock);

  CurBlock->getTerminator()->eraseFromParent();
  IRB.SetInsertPoint(CurBlock);
  {
    Value *Cond =
        IRB.CreateLoad(CondVar.Alloca->getAllocatedType(), CondVar.Alloca);
    IRB.CreateCondBr(Cond, ThenBlock, ExitBlock);
  }

  IRB.SetInsertPoint(ThenBlock);
  {
    IRB.CreateBr(ExitBlock);
  }

  IRB.SetInsertPoint(ExitBlock);
  {
    IRB.CreateBr(NextBlock);
  }

  IP = IRBuilderBase::InsertPoint(ThenBlock, ThenBlock->begin());
  IRB.restoreIP(IP);
  {
    Then();
  }

  Scopes.pop_back();
  IP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  IRB.restoreIP(IP);
}

template <typename ThenLambda, typename ElseLambda>
void FuncBase::IfElse(Var &CondVar, ThenLambda &&Then, ElseLambda &&Else,
                  const char *File, int Line) {
  Function *F = getFunction();
  BasicBlock *CurBlock = IP.getBlock();
  BasicBlock *NextBlock =
      CurBlock->splitBasicBlock(IP.getPoint(), CurBlock->getName() + ".split");

  auto ContIP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  Scopes.emplace_back(File, Line, ScopeKind::IF_ELSE, ContIP);

  BasicBlock *ThenBlock =
      BasicBlock::Create(F->getContext(), "if.then", F, NextBlock);
  BasicBlock *ElseBlock =
      BasicBlock::Create(F->getContext(), "if.else", F, NextBlock);
  BasicBlock *ExitBlock =
      BasicBlock::Create(F->getContext(), "if.cont", F, NextBlock);

  CurBlock->getTerminator()->eraseFromParent();
  IRB.SetInsertPoint(CurBlock);
  {
    Value *Cond =
        IRB.CreateLoad(CondVar.Alloca->getAllocatedType(), CondVar.Alloca);
    IRB.CreateCondBr(Cond, ThenBlock, ElseBlock);
  }

  IRB.SetInsertPoint(ThenBlock);
  {
    IRB.CreateBr(ExitBlock);
  }

  IRB.SetInsertPoint(ElseBlock);
  {
    IRB.CreateBr(ExitBlock);
  }

  IRB.SetInsertPoint(ExitBlock);
  {
    IRB.CreateBr(NextBlock);
  }

  IP = IRBuilderBase::InsertPoint(ThenBlock, ThenBlock->begin());
  IRB.restoreIP(IP);
  {
    Then();
  }
  IP = IRBuilderBase::InsertPoint(ElseBlock, ElseBlock->begin());
  IRB.restoreIP(IP);
  {
    Else();
  }

  IP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  IRB.restoreIP(IP);

  Scope S = Scopes.back();
  if (S.Kind != ScopeKind::IF_ELSE)
    PROTEUS_FATAL_ERROR("Syntax error, expected IF_ELSE end scope but "
                        "found unterminated scope " +
                        toString(S.Kind) + " @ " + S.File + ":" +
                        std::to_string(S.Line));
  Scopes.pop_back();
  IRB.restoreIP(S.ContIP);
}

template <typename RetT, typename... ArgT>
RetT Func<RetT, ArgT...>::operator()(ArgT... Args) {
  if (!J.isCompiled())
    J.compile();

  if (J.getTargetModel() != TargetModelType::HOST)
    PROTEUS_FATAL_ERROR(
        "Target is a GPU model, cannot directly run functions, use launch()");

  return Dispatch.run<RetT>(getName(), J.getObjectModuleRef(), Args...);
}

} // namespace proteus

#endif
