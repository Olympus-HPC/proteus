#ifndef PROTEUS_PASS_LAMBDA_ANALYSIS_H
#define PROTEUS_PASS_LAMBDA_ANALYSIS_H

namespace llvm {
class Module;
}

namespace proteus {

bool runLambdaAnalysis(llvm::Module &M, bool IsLTO);

} // namespace proteus

#endif
