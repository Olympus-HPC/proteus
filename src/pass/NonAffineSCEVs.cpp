#include "proteus/NonAffineSCEVs.hpp"
//#include "proteus/log.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

using namespace llvm;
//using namespace proteus;

#define DEBUG_TYPE "jitpass"
#ifdef PROTEUS_ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

namespace {
//REGISTER_LOG(console, DEBUG_TYPE);
} // namespace

namespace proteus {
namespace SCEVType {
/// @brief The type of a SCEV
///
/// To check for the validity of a SCEV we assign to each SCEV a type. The
/// possible types are INT, PARAM, IV and INVALID. The order of the types is
/// important. The subexpressions of SCEV with a type X can only have a type
/// that is smaller or equal than X.
enum TYPE {
  // An integer value.
  INT,

  // An expression that is constant during the execution of the Scop,
  // but that may depend on parameters unknown at compile time.
  PARAM,

  // An expression that may change during the execution of the SCoP.
  IV,

  // An invalid expression.
  INVALID
};
} // namespace SCEVType

/// @brief The result the validator returns for a SCEV expression.
class ValidatorResult {
  /// @brief The type of the expression
  SCEVType::TYPE Type;

  /// @brief The set of Parameters in the expression.
  SmallPtrSet<const SCEV *, 2> Parameters;

public:
  /// @brief The copy constructor
  ValidatorResult(const ValidatorResult &Source) {
    Type = Source.Type;
    Parameters = Source.Parameters;
  }

  /// @brief Construct a result with a certain type and no parameters.
  ValidatorResult(SCEVType::TYPE Type) : Type(Type) {
    assert(Type != SCEVType::PARAM && "Did you forget to pass the parameter");
  }

  /// @brief Construct a result with a certain type and a single parameter.
  ValidatorResult(SCEVType::TYPE Type, const SCEV *Expr) : Type(Type) {
    Parameters.insert(Expr);
  }

  /// @brief Get the type of the ValidatorResult.
  SCEVType::TYPE getType() { return Type; }

  /// @brief Is the analyzed SCEV constant during the execution of the SCoP.
  bool isConstant() { return Type == SCEVType::INT || Type == SCEVType::PARAM; }

  /// @brief Is the analyzed SCEV valid.
  bool isValid() { return Type != SCEVType::INVALID; }

  /// @brief Is the analyzed SCEV of Type IV.
  bool isIV() { return Type == SCEVType::IV; }

  /// @brief Is the analyzed SCEV of Type INT.
  bool isINT() { return Type == SCEVType::INT; }

  /// @brief Is the analyzed SCEV of Type PARAM.
  bool isPARAM() { return Type == SCEVType::PARAM; }

  /// @brief Get the parameters of this validator result.
  SmallPtrSet<const SCEV *, 2> &getParameters() { return Parameters; }

  /// @brief Add the parameters of Source to this result.
  void addParamsFrom(const ValidatorResult &Source) {
    for (auto *P : Source.Parameters) {
      Parameters.insert(P);
    }
  }

  /// @brief Merge a result.
  ///
  /// This means to merge the parameters and to set the Type to the most
  /// specific Type that matches both.
  void merge(const ValidatorResult &ToMerge) {
    Type = std::max(Type, ToMerge.Type);
    addParamsFrom(ToMerge);
  }

  void print(raw_ostream &OS) {
    switch (Type) {
    case SCEVType::INT:
      OS << "SCEVType::INT";
      break;
    case SCEVType::PARAM:
      OS << "SCEVType::PARAM";
      break;
    case SCEVType::IV:
      OS << "SCEVType::IV";
      break;
    case SCEVType::INVALID:
      OS << "SCEVType::INVALID";
      break;
    }
  }
};

raw_ostream &operator<<(raw_ostream &OS, class ValidatorResult &VR) {
  VR.print(OS);
  return OS;
}

class NonAffSCEVValidator
    : public SCEVVisitor<NonAffSCEVValidator, class ValidatorResult> {
private:
  const Region *R;
  Loop *Scope;
  ScalarEvolution &SE;
  polly::InvariantLoadsSetTy *ILS;

public:
  NonAffSCEVValidator(const Region *R, Loop *Scope, ScalarEvolution &SE,
                      polly::InvariantLoadsSetTy *ILS)
      : R(R), Scope(Scope), SE(SE), ILS(ILS) {}

  class ValidatorResult visitConstant(const SCEVConstant *Constant) {
    return ValidatorResult(SCEVType::INT);
  }

  class ValidatorResult visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    return visit(Expr->getOperand());
  }

  class ValidatorResult visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    ValidatorResult Op = visit(Expr->getOperand());

    switch (Op.getType()) {
    case SCEVType::INT:
    case SCEVType::PARAM:
      // We currently do not represent a truncate expression as an affine
      // expression. If it is constant during Scop execution, we treat it as a
      // parameter.
      return ValidatorResult(SCEVType::PARAM, Expr);
    case SCEVType::IV:
      DEBUG(dbgs() << "INVALID: ZeroExtend of SCEVType::IV expression");
      return ValidatorResult(SCEVType::INVALID);
    case SCEVType::INVALID:
      return Op;
    }

    llvm_unreachable("Unknown SCEVType");
  }

  class ValidatorResult visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return visit(Expr->getOperand());
  }

  class ValidatorResult visitAddExpr(const SCEVAddExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int I = 0, E = Expr->getNumOperands(); I < E; ++I) {
      ValidatorResult Op = visit(Expr->getOperand(I));
      Return.merge(Op);

      // Early exit.
      if (!Return.isValid())
        break;
    }

    // TODO: Check for NSW and NUW.
    return Return;
  }

  class ValidatorResult visitMulExpr(const SCEVMulExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    bool HasMultipleParams = false;

    for (int I = 0, E = Expr->getNumOperands(); I < E; ++I) {
      ValidatorResult Op = visit(Expr->getOperand(I));

      if (Op.isINT())
        continue;

      if (Op.isPARAM() && Return.isPARAM()) {
        HasMultipleParams = true;
        continue;
      }

      if ((Op.isIV() || Op.isPARAM()) && !Return.isINT()) {
        DEBUG(dbgs() << "INVALID: More than one non-int operand in MulExpr\n"
                     << "\tExpr: " << *Expr << "\n"
                     << "\tPrevious expression type: " << Return << "\n"
                     << "\tNext operand (" << Op
                     << "): " << *Expr->getOperand(I) << "\n");

        return ValidatorResult(SCEVType::INVALID);
      }

      Return.merge(Op);
    }

    if (HasMultipleParams && Return.isValid())
      return ValidatorResult(SCEVType::PARAM, Expr);

    // TODO: Check for NSW and NUW.
    return Return;
  }

  class ValidatorResult visitUDivExpr(const SCEVUDivExpr *Expr) {
    ValidatorResult LHS = visit(Expr->getLHS());
    ValidatorResult RHS = visit(Expr->getRHS());

    // We currently do not represent an unsigned division as an affine
    // expression. If the division is constant during Scop execution we treat it
    // as a parameter, otherwise we bail out.
    if (LHS.isConstant() && RHS.isConstant())
      return ValidatorResult(SCEVType::PARAM, Expr);

    // As long as unsigned division is not represented as an affine expression,
    // the JIT can compute the result from the params at run-time.
    if (LHS.isPARAM() || RHS.isPARAM())
      return ValidatorResult(SCEVType::PARAM, Expr);

    DEBUG(dbgs() << "INVALID: unsigned division of non-constant expressions");
    return ValidatorResult(SCEVType::INVALID);
  }

  class ValidatorResult visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    ValidatorResult Start = visit(Expr->getStart());
    ValidatorResult Recurrence = visit(Expr->getStepRecurrence(SE));

    if (!Start.isValid())
      return Start;

    if (!Recurrence.isValid())
      return Recurrence;

    auto *L = Expr->getLoop();
    if (R->contains(L) && (!Scope || !L->contains(Scope))) {
      DEBUG(dbgs() << "INVALID: AddRec out of a loop whose exit value is not "
                      "synthesizable");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (R->contains(L)) {
      if (Recurrence.isINT()) {
        ValidatorResult Result(SCEVType::IV);
        Result.addParamsFrom(Start);
        return Result;
      }

      if (Recurrence.isPARAM()) {
        ValidatorResult Result(SCEVType::IV);
        Result.addParamsFrom(Start);
        Result.addParamsFrom(Recurrence);

        DEBUG(dbgs() << "VALID: AddRec within scop has parametrized"
                     << " recurrence part\n");
        return Result;
      }

      DEBUG(dbgs() << "INVALID: AddRec within scop has non-int"
                      "recurrence part");
      return ValidatorResult(SCEVType::INVALID);
    }

    assert(Start.isConstant() && Recurrence.isConstant() &&
           "Expected 'Start' and 'Recurrence' to be constant");

    // Directly generate ValidatorResult for Expr if 'start' is zero.
    if (Expr->getStart()->isZero())
      return ValidatorResult(SCEVType::PARAM, Expr);

    // Translate AddRecExpr from '{start, +, inc}' into 'start + {0, +, inc}'
    // if 'start' is not zero.
    const SCEV *ZeroStartExpr = SE.getAddRecExpr(
        SE.getConstant(Expr->getStart()->getType(), 0),
        Expr->getStepRecurrence(SE), Expr->getLoop(), Expr->getNoWrapFlags());

    ValidatorResult ZeroStartResult =
        ValidatorResult(SCEVType::PARAM, ZeroStartExpr);
    ZeroStartResult.addParamsFrom(Start);

    return ZeroStartResult;
  }

  class ValidatorResult visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int I = 0, E = Expr->getNumOperands(); I < E; ++I) {
      ValidatorResult Op = visit(Expr->getOperand(I));

      if (!Op.isValid())
        return Op;

      Return.merge(Op);
    }

    return Return;
  }

  class ValidatorResult visitUMaxExpr(const SCEVUMaxExpr *Expr) {
    // We do not support unsigned operations. If 'Expr' is constant during Scop
    // execution we treat this as a parameter, otherwise we bail out.
    for (int I = 0, E = Expr->getNumOperands(); I < E; ++I) {
      ValidatorResult Op = visit(Expr->getOperand(I));

      if (!Op.isConstant()) {
        DEBUG(dbgs() << "INVALID: UMaxExpr has a non-constant operand");
        return ValidatorResult(SCEVType::INVALID);
      }
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }

  ValidatorResult visitGenericInst(Instruction *I, const SCEV *S) {
    if (R->contains(I)) {
      DEBUG(dbgs() << "INVALID: UnknownExpr references an instruction "
                      "within the region\n");
      DEBUG(dbgs() << *I << "\n");
      return ValidatorResult(SCEVType::INVALID);
    }

    return ValidatorResult(SCEVType::PARAM, S);
  }

  ValidatorResult visitLoadInstruction(Instruction *I, const SCEV *S) {
    if (R->contains(I) && ILS) {
      ILS->insert(cast<LoadInst>(I));
      return ValidatorResult(SCEVType::PARAM, S);
    }

    return visitGenericInst(I, S);
  }

  ValidatorResult visitDivision(const SCEV *Dividend, const SCEV *Divisor,
                                const SCEV *DivExpr,
                                Instruction *SDiv = nullptr) {
    // First check if we might be able to model the division, thus if the
    // divisor is constant. If so, check the dividend, otherwise check if
    // the whole division can be seen as a parameter.
    if (isa<SCEVConstant>(Divisor) && !Divisor->isZero())
      return visit(Dividend);

    // For signed divisions use the SDiv instruction to check for a parameter
    // division, for unsigned divisions check the operands.
    if (SDiv)
      return visitGenericInst(SDiv, DivExpr);

    ValidatorResult LHS = visit(Dividend);
    ValidatorResult RHS = visit(Divisor);
    if (LHS.isConstant() && RHS.isConstant())
      return ValidatorResult(SCEVType::PARAM, DivExpr);

    DEBUG(dbgs() << "INVALID: unsigned division of non-constant expressions");
    return ValidatorResult(SCEVType::INVALID);
  }

  ValidatorResult visitSDivInstruction(Instruction *SDiv, const SCEV *Expr) {
    assert(SDiv->getOpcode() == Instruction::SDiv &&
           "Assumed SDiv instruction!");

    auto *Dividend = SE.getSCEV(SDiv->getOperand(0));
    auto *Divisor = SE.getSCEV(SDiv->getOperand(1));
    return visitDivision(Dividend, Divisor, Expr, SDiv);
  }

  ValidatorResult visitSRemInstruction(Instruction *SRem, const SCEV *S) {
    assert(SRem->getOpcode() == Instruction::SRem &&
           "Assumed SRem instruction!");

    auto *Divisor = SRem->getOperand(1);
    auto *CI = dyn_cast<ConstantInt>(Divisor);
    if (!CI || CI->isZeroValue())
      return visitGenericInst(SRem, S);

    auto *Dividend = SRem->getOperand(0);
    auto *DividendSCEV = SE.getSCEV(Dividend);
    return visit(DividendSCEV);
  }

  ValidatorResult visitUnknown(const SCEVUnknown *Expr) {
    Value *V = Expr->getValue();

    if (!Expr->getType()->isIntegerTy()) {
      DEBUG(dbgs() << "INVALID: UnknownExpr is not an integer");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (isa<UndefValue>(V)) {
      DEBUG(dbgs() << "INVALID: UnknownExpr references an undef value");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (Instruction *I = dyn_cast<Instruction>(Expr->getValue())) {
      switch (I->getOpcode()) {
      case Instruction::IntToPtr:
        return visit(SE.getSCEVAtScope(I->getOperand(0), Scope));
      case Instruction::PtrToInt:
        return visit(SE.getSCEVAtScope(I->getOperand(0), Scope));
      case Instruction::Load:
        return visitLoadInstruction(I, Expr);
      case Instruction::SDiv:
        return visitSDivInstruction(I, Expr);
      case Instruction::SRem:
        return visitSRemInstruction(I, Expr);
      default:
        return visitGenericInst(I, Expr);
      }
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }
};

bool isNonAffineExpr(const Region *R, llvm::Loop *Scope, const SCEV *Expr,
                     ScalarEvolution &SE, polly::InvariantLoadsSetTy *ILS) {
  std::string Buf;
  raw_string_ostream Os(Buf);
  Os << "\n";
  Os << "Expr: " << *Expr << "\n";
  if (isa<SCEVCouldNotCompute>(Expr))
    return false;

  NonAffSCEVValidator Validator(R, Scope, SE, ILS);
  Os << "Region: " << R->getNameStr() << "\n";
  Os << " -> ";

  ValidatorResult Result = Validator.visit(Expr);

  if (Result.isValid())
    Os << "VALID\n";
  Os << "\n";
  DEBUG(console->debug(Os.str()));

  return Result.isValid();
}

std::vector<const SCEV *> getParamsInNonAffineExpr(const Region *R, Loop *Scope,
                                                   const SCEV *Expr,
                                                   ScalarEvolution &SE) {
  if (isa<SCEVCouldNotCompute>(Expr))
    return std::vector<const SCEV *>();

  polly::InvariantLoadsSetTy ILS;
  NonAffSCEVValidator Validator(R, Scope, SE, &ILS);
  ValidatorResult Result = Validator.visit(Expr);
  assert(Result.isValid() && "Requested parameters for an invalid SCEV!");

  std::vector<const SCEV *> Res;
  for (auto *P : Result.getParameters()) {
    Res.push_back(P);
  }
  return Res;
}
} // namespace proteus
