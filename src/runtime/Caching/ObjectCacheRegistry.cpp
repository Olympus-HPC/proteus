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

namespace proteus {

ObjectCacheRegistry &ObjectCacheRegistry::instance() {
  static auto *Registry = new ObjectCacheRegistry();
  return *Registry;
}

void ObjectCacheRegistry::create(StringRef Label) {
  std::string Key = Label.str();
  auto It = Chains.find(Key);
  if (It == Chains.end()) {
    Chains.emplace(Key, std::make_unique<ObjectCacheChain>(Key));
  }
}

std::optional<std::reference_wrapper<ObjectCacheChain>>
ObjectCacheRegistry::get(StringRef Label) {
  auto It = Chains.find(Label.str());
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
