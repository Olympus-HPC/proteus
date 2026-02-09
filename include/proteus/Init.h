//===-- Init.h -- Proteus initialization state --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_INIT_H
#define PROTEUS_INIT_H

namespace proteus {

bool &proteusIsInitialized();
void ensureProteusInitialized();

void init();
void finalize();
void enable();
void disable();

} // namespace proteus

#endif
