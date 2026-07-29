#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#if __has_include(<llvm/Plugins/PassPlugin.h>)
#include <llvm/Plugins/PassPlugin.h>
#elif __has_include(<llvm/Passes/PassPlugin.h>)
#include <llvm/Passes/PassPlugin.h>
#else
#error "Cannot find LLVM PassPlugin.h"
#endif
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <utility>

namespace {

class JITTestPass : public llvm::PassInfoMixin<JITTestPass> {
public:
  explicit JITTestPass(std::string PipelineName)
      : PipelineName(std::move(PipelineName)) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    llvm::outs() << "[JITTestPass] " << PipelineName << " " << M.getName()
                 << "\n";
    return llvm::PreservedAnalyses::all();
  }

private:
  std::string PipelineName;
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  llvm::outs() << "[JITTestPluginInfo]\n";
  return {LLVM_PLUGIN_API_VERSION, "JITTestPass", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (!Name.starts_with("jit-test-pass"))
                    return false;
                  MPM.addPass(JITTestPass(Name.str()));
                  return true;
                });
          }};
}
