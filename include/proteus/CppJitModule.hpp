#ifndef PROTEUS_CPPFRONTEND_HPP
#define PROTEUS_CPPFRONTEND_HPP

#include <llvm/Support/Debug.h>

#include "proteus/Frontend/Dispatcher.hpp"

namespace proteus {

class CppJitModule {
private:
  TargetModelType TargetModel;
  std::string Code;
  HashT ModuleHash;
  std::vector<std::string> ExtraArgs;

  Dispatcher &Dispatch;
  std::unique_ptr<MemoryBuffer> ObjectModule;
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
         << ((TargetModel != TargetModelType::HOST) ? "__global__ " : "")
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

    template <typename... ArgT>
    auto launch(LaunchDims GridDim, LaunchDims BlockDim, uint64_t ShmemSize,
                void *Stream, ArgT... Args) {
      if (!InstanceModule) {
        std::string InstanceCode = buildCode<void, ArgT...>();
        InstanceModule =
            std::make_unique<CppJitModule>(TargetModel, InstanceCode);
        InstanceModule->compile();
      }

      void *Ptrs[sizeof...(ArgT)] = {(void *)&Args...};

      return InstanceModule->Dispatch.launch(
          EntryFuncName, GridDim, BlockDim, Ptrs, ShmemSize, Stream,
          InstanceModule->getObjectModuleRef());
    }

    template <typename RetOrSig, typename... ArgT> auto run(ArgT &&...Args) {
      static_assert(!std::is_function_v<RetOrSig>,
                    "Function signature type is not yet supported");

      if (!InstanceModule) {
        std::string InstanceCode = buildCode<RetOrSig, ArgT...>();
        InstanceModule =
            std::make_unique<CppJitModule>(TargetModel, InstanceCode);
        InstanceModule->compile();
      }

      return InstanceModule->Dispatch.run<RetOrSig, ArgT...>(
          EntryFuncName, InstanceModule->getObjectModuleRef(),
          std::forward<ArgT>(Args)...);
    }
  };

  struct CompilationResult {
    // Declare Ctx first to ensure it is destroyed after Mod.
    std::unique_ptr<LLVMContext> Ctx;
    std::unique_ptr<Module> Mod;
  };

protected:
  CompilationResult compileCppToIR();

public:
  explicit CppJitModule(TargetModelType TargetModel, StringRef Code,
                        const std::vector<std::string> &ExtraArgs = {});
  explicit CppJitModule(StringRef Target, StringRef Code,
                        const std::vector<std::string> &ExtraArgs = {});

  void compile();

  std::optional<MemoryBufferRef> getObjectModuleRef() const {
    if (!ObjectModule)
      return std::nullopt;

    return ObjectModule->getMemBufferRef();
  }

  template <typename RetOrSig, typename... ArgT>
  auto run(const char *FuncName, ArgT &&...Args) {
    if (!IsCompiled)
      compile();

    // TODO: We could cache the function address if we have a stateful object to
    // store and return to the user for invocations, similar to DSL
    // KernelHandle.
    return Dispatch.run<RetOrSig, ArgT...>(FuncName, getObjectModuleRef(),
                                           std::forward<ArgT>(Args)...);
  }

  template <typename... ArgT>
  auto launch(StringRef KernelName, LaunchDims GridDim, LaunchDims BlockDim,
              uint64_t ShmemSize, void *Stream, ArgT... Args) {
    if (!IsCompiled)
      compile();

    void *Ptrs[sizeof...(ArgT)] = {(void *)&Args...};

    // TODO: We could cache the function address if we have a stateful object to
    // store and return to the user for invocations, similar to DSL
    // KernelHandle.
    return Dispatch.launch(KernelName, GridDim, BlockDim, Ptrs, ShmemSize,
                           Stream, getObjectModuleRef());
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
};

} // namespace proteus

#endif
