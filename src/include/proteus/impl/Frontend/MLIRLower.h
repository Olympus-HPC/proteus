#ifndef PROTEUS_IMPL_FRONTEND_MLIRLOWER_H
#define PROTEUS_IMPL_FRONTEND_MLIRLOWER_H

#include "proteus/Frontend/TargetModel.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <memory>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace proteus {

struct MLIRLoweringOptions {
  TargetModelType TargetModel = TargetModelType::HOST;
  std::string DeviceArch;
  int OptLevel = 3;
  std::string TargetTriple;
  std::string Features;
  std::string DiagnosticPrefix = "MLIRLower";
};

struct MLIRLoweringResult {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Mod;
};

void registerMLIRLoweringDialects(mlir::DialectRegistry &Registry);
void loadMLIRLoweringDialects(mlir::MLIRContext &Context);

MLIRLoweringResult lowerMLIRModuleToLLVM(mlir::ModuleOp Module,
                                         const MLIRLoweringOptions &Options);

} // namespace proteus

#endif // PROTEUS_IMPL_FRONTEND_MLIRLOWER_H
