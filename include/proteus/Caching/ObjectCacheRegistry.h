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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace proteus {

class ObjectCacheChain;

class ObjectCacheRegistry {
public:
  static ObjectCacheRegistry &instance();

  ~ObjectCacheRegistry();

  void create(const std::string &Label);

  std::optional<std::reference_wrapper<ObjectCacheChain>>
  get(const std::string &Label);

  void printStats();

private:
  ObjectCacheRegistry() = default;
  ObjectCacheRegistry(const ObjectCacheRegistry &) = delete;
  ObjectCacheRegistry &operator=(const ObjectCacheRegistry &) = delete;

  std::unordered_map<std::string, std::unique_ptr<ObjectCacheChain>> Chains;
};

} // namespace proteus

#endif
