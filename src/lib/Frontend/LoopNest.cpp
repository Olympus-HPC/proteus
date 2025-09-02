#include "proteus/Frontend/LoopNest.hpp"

namespace proteus {

LoopBoundInfo::LoopBoundInfo(Var &IterVar, Var &Init,
                                             Var &UpperBound, Var &Inc)
    : IterVar(IterVar), Init(Init), UpperBound(UpperBound), Inc(Inc) {}
} // namespace proteus
