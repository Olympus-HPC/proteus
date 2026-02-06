//===-- ObjectCacheRegistry.h -- Object cache chain registry header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_OBJECTCACHEREGISTRY_H
#define PROTEUS_OBJECTCACHEREGISTRY_H

#include "proteus/Caching/ObjectCacheChain.h"

#include <llvm/ADT/StringRef.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace proteus {

using namespace llvm;

class ObjectCacheRegistry {
public:
  static ObjectCacheRegistry &instance();

  void create(StringRef Label);

  std::optional<std::reference_wrapper<ObjectCacheChain>> get(StringRef Label);

  void finalizeAll();

  void printStatsAll();

private:
  ObjectCacheRegistry() = default;
  ObjectCacheRegistry(const ObjectCacheRegistry &) = delete;
  ObjectCacheRegistry &operator=(const ObjectCacheRegistry &) = delete;

  std::unordered_map<std::string, std::unique_ptr<ObjectCacheChain>> Chains;
};

} // namespace proteus

#endif
