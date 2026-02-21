#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include <iostream>

namespace {
class TestAction : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override {
    std::cout << "[cout] ==> Clang Plugin running\n";
    return std::make_unique<clang::ASTConsumer>();
  }
  bool ParseArgs(const clang::CompilerInstance &,
                 const std::vector<std::string> &) override {
    return true;
  }

  // ADD THIS METHOD to make it run automatically.
  PluginASTAction::ActionType getActionType() override {
    // Run in addition to the main action.
    return AddBeforeMainAction;
  }
};
} // namespace

static clang::FrontendPluginRegistry::Add<TestAction> X("test-plugin",
                                                        "Test plugin");
