//===-- ObjectCacheRegistry.cpp -- Centralized cache ownership ------===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/Caching/ObjectCacheRegistry.h"

namespace proteus {

ObjectCacheRegistry &ObjectCacheRegistry::instance() {
  static auto *Registry = new ObjectCacheRegistry();
  return *Registry;
}

ObjectCacheChain &ObjectCacheRegistry::get(const std::string &Label) {
  auto &Chain = Chains[Label];
  if (!Chain)
    Chain = std::make_unique<ObjectCacheChain>(Label);
  return *Chain;
}

void ObjectCacheRegistry::finalize() {
  for (auto &[Label, Chain] : Chains)
    Chain->finalize();
}

void ObjectCacheRegistry::printStats() {
  for (auto &[Label, Chain] : Chains)
    Chain->printStats();
}

} // namespace proteus
