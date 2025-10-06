#ifndef PROTEUS_CPPFRONTEND_HPP
#define PROTEUS_CPPFRONTEND_HPP

#include <llvm/Support/Debug.h>

#include "proteus/CompiledLibrary.hpp"
#include "proteus/Frontend/Dispatcher.hpp"

namespace proteus {

class CppJitModule {
private:
  TargetModelType TargetModel;
  std::string Code;
  HashT ModuleHash;
  std::vector<std::string> ExtraArgs;

  // Optimization level used when emitting IR.
  static constexpr const char *FrontendOptLevel = "O3";

  Dispatcher &Dispatch;
  std::unique_ptr<CompiledLibrary> Library = nullptr;
  bool IsCompiled = false;

  // TODO: We don't cache CodeInstances so if a user re-creates the exact same
  // instantiation it will create a new CodeInstance. This creation cost is
  // mitigated because the dispatcher caches the compiled object so we will pay
  // the overhead for building the instantiation code but not for compilation.
  // Nevertheless, we return the CodeInstance object when created, so the user
  // can avoid any re-creation overhead by using the returned object to run or
  // launch. Re-think caching and dispatchers, and on more restrictive
  // interfaces.
  struct CodeInstance {
    TargetModelType TargetModel;
    StringRef TemplateCode;
    std::string InstanceName;
    std::unique_ptr<CppJitModule> InstanceModule;
    std::string EntryFuncName;
    void *FuncPtr = nullptr;

    CodeInstance(TargetModelType TargetModel, StringRef TemplateCode,
                 StringRef InstanceName)
        : TargetModel(TargetModel), TemplateCode(TemplateCode),
          InstanceName(InstanceName) {
      EntryFuncName = "__jit_instance_" + this->InstanceName;
      // Replace characters '<', '>', ',' with $ to create a unique for the
      // entry function.
      std::replace_if(
          EntryFuncName.begin(), EntryFuncName.end(),
          [](char C) { return C == '<' || C == '>' || C == ','; }, '$');
    }

    // Compile-time type name (no RTTI).
    template <class T> constexpr std::string_view typeName() {
      // Apparently we are more interested in clang, but leaving the others for
      // completeness.
#if defined(__clang__)
      // "std::string_view type_name() [T = int]"
      std::string_view P = __PRETTY_FUNCTION__;
      auto B = P.find("[T = ") + 5;
      auto E = P.rfind(']');
      return P.substr(B, E - B);
#elif defined(__GNUC__)
      // "... with T = int; ..."
      std::string_view P = __PRETTY_FUNCTION__;
      auto B = P.find("with T = ") + 9;
      auto E = P.find(';', B);
      return P.substr(B, E - B);
#elif defined(_MSC_VER)
      // "std::string_view __cdecl type_name<int>(void)"
      std::string_view P = __FUNCSIG__;
      auto B = P.find("type_name<") + 10;
      auto E = P.find(">(void)", B);
      return P.substr(B, B - E);
#else
      PROTEUS_FATAL_ERROR("Unsupported compiler");
#endif
    }

    template <typename RetT, typename... ArgT, std::size_t... I>
    std::string buildFunctionEntry(std::index_sequence<I...>) {
      SmallString<256> FuncS;
      raw_svector_ostream OS(FuncS);

      OS << "extern \"C\" " << typeName<RetT>() << " "
         << ((!isHostTargetModel(TargetModel)) ? "__global__ " : "")
         << EntryFuncName << "(";
      ((OS << (I ? ", " : "")
           << typeName<
                  std::decay_t<std::tuple_element_t<I, std::tuple<ArgT...>>>>()
           << " Arg" << I),
       ...);
      OS << ')';

      std::string ArgList;
      ((ArgList += (I == 0 ? "" : ", ") + ("Arg" + std::to_string(I))), ...);
      OS << "{ ";
      if constexpr (!std::is_void_v<RetT>) {
        OS << "return ";
      }
      OS << InstanceName << "(";
      ((OS << (I == 0 ? "" : ", ") << "Arg" << std::to_string(I)), ...);
      OS << "); }";

      return std::string(FuncS);
    }

    template <typename RetOrSig, typename... ArgT> std::string buildCode() {
      std::string FunctionCode = buildFunctionEntry<RetOrSig, ArgT...>(
          std::index_sequence_for<ArgT...>{});

      auto ReplaceAll = [](std::string &S, std::string_view From,
                           std::string_view To) {
        if (From.empty())
          return;
        std::size_t Pos = 0;
        while ((Pos = S.find(From, Pos)) != std::string::npos) {
          S.replace(Pos, From.size(), To);
          // Skip over the just-inserted text.
          Pos += To.size();
        }
      };

      std::string InstanceCode = TemplateCode.str();
      // Demote kernels to device function to call the templated instance from
      // the entry function.
      ReplaceAll(InstanceCode, "__global__", "__device__");
      InstanceCode = InstanceCode + FunctionCode;

      return InstanceCode;
    }

    template <typename RetT, typename... ArgT> void compile() {
      std::string InstanceCode = buildCode<RetT, ArgT...>();
      InstanceModule =
          std::make_unique<CppJitModule>(TargetModel, InstanceCode);
      InstanceModule->compile();

      FuncPtr = InstanceModule->Dispatch.getFunctionAddress(
          EntryFuncName, InstanceModule->ModuleHash,
          InstanceModule->getLibrary());
    }

    template <typename... ArgT>
    auto launch(LaunchDims GridDim, LaunchDims BlockDim, uint64_t ShmemSize,
                void *Stream, ArgT... Args) {
      if (!InstanceModule) {
        compile<void, ArgT...>();
      }

      void *Ptrs[sizeof...(ArgT)] = {(void *)&Args...};

      return InstanceModule->Dispatch.launch(FuncPtr, GridDim, BlockDim, Ptrs,
                                             ShmemSize, Stream);
    }

    template <typename RetOrSig, typename... ArgT>
    RetOrSig run(ArgT &&...Args) {
      static_assert(!std::is_function_v<RetOrSig>,
                    "Function signature type is not yet supported");

      if (!InstanceModule) {
        compile<RetOrSig, ArgT...>();
      }

      if constexpr (std::is_void_v<RetOrSig>)
        InstanceModule->Dispatch.run<RetOrSig(ArgT...)>(FuncPtr, Args...);
      else
        return InstanceModule->Dispatch.run<RetOrSig(ArgT...)>(FuncPtr,
                                                               Args...);
    }
  };

  struct CompilationResult {
    // Declare Ctx first to ensure it is destroyed after Mod.
    std::unique_ptr<LLVMContext> Ctx = nullptr;
    std::unique_ptr<Module> Mod = nullptr;
  };

protected:
  CompilationResult compileCppToIR();
  void compileCppToDynamicLibrary();

public:
  explicit CppJitModule(TargetModelType TargetModel, StringRef Code,
                        const std::vector<std::string> &ExtraArgs = {});
  explicit CppJitModule(StringRef Target, StringRef Code,
                        const std::vector<std::string> &ExtraArgs = {});

  void compile();

  CompiledLibrary &getLibrary() {
    if (!IsCompiled)
      compile();

    if (!Library)
      PROTEUS_FATAL_ERROR("Expected non-null library after compilation");

    return *Library;
  }

  template <typename... ArgT>
  auto instantiate([[maybe_unused]] StringRef FuncName,
                   [[maybe_unused]] ArgT... Args) {
    std::string InstanceName = FuncName.str() + "<";
    bool First = true;
    ((InstanceName +=
      (First ? "" : ",") + std::string(std::forward<ArgT>(Args)),
      First = false),
     ...);

    InstanceName += ">";

    return CodeInstance{TargetModel, Code, InstanceName};
  }

  template <typename Sig> struct FunctionHandle;
  template <typename RetT, typename... ArgT>
  struct FunctionHandle<RetT(ArgT...)> {
    CppJitModule &M;
    void *FuncPtr;
    explicit FunctionHandle(CppJitModule &M, void *FuncPtr)
        : M(M), FuncPtr(FuncPtr) {}

    RetT run(ArgT... Args) {
      if constexpr (std::is_void_v<RetT>) {
        M.Dispatch.template run<RetT(ArgT...)>(FuncPtr,
                                               std::forward<ArgT>(Args)...);
      } else {
        return M.Dispatch.template run<RetT(ArgT...)>(
            FuncPtr, std::forward<ArgT>(Args)...);
      }
    }
  };

  template <typename Sig> struct KernelHandle;
  template <typename RetT, typename... ArgT>
  struct KernelHandle<RetT(ArgT...)> {
    CppJitModule &M;
    void *FuncPtr = nullptr;
    explicit KernelHandle(CppJitModule &M, void *FuncPtr)
        : M(M), FuncPtr(FuncPtr) {
      static_assert(std::is_void_v<RetT>, "Kernel function must return void");
    }

    auto launch(LaunchDims GridDim, LaunchDims BlockDim, uint64_t ShmemSize,
                void *Stream, ArgT... Args) {
      void *Ptrs[sizeof...(ArgT)] = {(void *)&Args...};

      return M.Dispatch.launch(FuncPtr, GridDim, BlockDim, Ptrs, ShmemSize,
                               Stream);
    }
  };
  template <typename Sig> FunctionHandle<Sig> getFunction(StringRef Name) {
    if (!IsCompiled)
      compile();

    if (!isHostTargetModel(TargetModel))
      PROTEUS_FATAL_ERROR("Error: getFunction() applies only to host modules");

    void *FuncPtr = Dispatch.getFunctionAddress(Name, ModuleHash, getLibrary());

    return FunctionHandle<Sig>(*this, FuncPtr);
  }

  template <typename Sig> KernelHandle<Sig> getKernel(StringRef Name) {
    if (!IsCompiled)
      compile();

    if (TargetModel == TargetModelType::HOST)
      PROTEUS_FATAL_ERROR("Error: getKernel() applies only to device modules");

    void *FuncPtr = Dispatch.getFunctionAddress(Name, ModuleHash, getLibrary());

    return KernelHandle<Sig>(*this, FuncPtr);
  }
};

} // namespace proteus

#endif
