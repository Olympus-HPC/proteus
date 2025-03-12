/* SCEVValidators from Polly modified for PolyJIT.
*
* Copyright © 2016 Andreas Simbürger <simbuerg@lairosiel.de>
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
* DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
* OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#ifndef POLLI_NONAFFINESCEVS_H
#define POLLI_NONAFFINESCEVS_H

#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/SetVector.h"
#include <vector>

namespace llvm {
class Region;
class SCEV;
class SCEVConstant;
class ScalarEvolution;
class Value;
class Loop;
class LoadInst;
} // namespace llvm

namespace polli {
bool isNonAffineExpr(const llvm::Region *R, llvm::Loop *Scope,
                     const llvm::SCEV *Expr, llvm::ScalarEvolution &SE,
                     polly::InvariantLoadsSetTy *ILS);

std::vector<const llvm::SCEV *>
getParamsInNonAffineExpr(const llvm::Region *R, llvm::Loop *Scope,
                         const llvm::SCEV *Expr, llvm::ScalarEvolution &SE);
} // namespace polli
#endif // POLLI_NONAFFINESCEVS_H
