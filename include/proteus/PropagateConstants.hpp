#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#ifndef PROTEUS_PROPAGATE_CONSTANTS_HPP
#define PROTEUS_PROPAGATE_CONSTANTS_HPP

// Optional: ADCE (dead code cleanup)
#include "llvm/Transforms/Scalar/ADCE.h"
namespace proteus {
class PropagateConstants {
public:
static void runConstantPropagation(llvm::Module &M) {
    llvm::PassBuilder PB;

    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    // Register required analyses
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Function-level passes
    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::SCCPPass());
    FPM.addPass(llvm::InstCombinePass());
    FPM.addPass(llvm::ADCEPass());  // Optional but helps remove unused code

    // Module-level pass to apply FPM
    llvm::ModulePassManager MPM;
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));

    // Run the pipeline
    MPM.run(M, MAM);
}
};
} // namespace proteus

#endif
