//===-- ObjectCacheRegistry.cpp -- Object cache chain registry impl --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/Caching/ObjectCacheRegistry.h"
#include "proteus/Caching/ObjectCacheChain.h"

namespace proteus {

ObjectCacheRegistry::~ObjectCacheRegistry() = default;

ObjectCacheRegistry &ObjectCacheRegistry::instance() {
  static auto *Registry = new ObjectCacheRegistry();
  return *Registry;
}

void ObjectCacheRegistry::create(const std::string &Label) {
  auto It = Chains.find(Label);
  if (It == Chains.end()) {
    Chains.emplace(Label, std::make_unique<ObjectCacheChain>(Label));
  }
}

std::optional<std::reference_wrapper<ObjectCacheChain>>
ObjectCacheRegistry::get(const std::string &Label) {
  auto It = Chains.find(Label);
  if (It == Chains.end()) {
    return std::nullopt;
  }
  return std::ref(*It->second);
}

void ObjectCacheRegistry::printStats() {
  for (auto &[_, Chain] : Chains) {
    Chain->printStats();
  }
}

} // namespace proteus
