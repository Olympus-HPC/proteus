#ifndef PROTEUS_FRONTEND_KERNEL_NAME_H
#define PROTEUS_FRONTEND_KERNEL_NAME_H

#include <optional>
#include <string>
#include <utility>

namespace llvm {
class StringRef;
} // namespace llvm

namespace proteus {

class HashT;

// KernelName pairs the base symbol of a function with an optional suffix
// derived from its hash.
class KernelName {
private:
  std::string Base;
  // By storing the mangled suffix, HashT can remain incomplete.
  std::optional<std::string> HashSuffix;

public:
  KernelName(std::string Base) : Base(std::move(Base)) {}
  KernelName(const char *Base) : Base(Base) {}
  KernelName(const llvm::StringRef &Base);
  KernelName(std::string Base, const HashT &Hash);

  const std::string &base() const { return Base; }

  std::string suffix() const {
    return HashSuffix ? *HashSuffix : std::string{};
  }

  std::string mangled() const { return Base + suffix(); }

  bool hasSuffix() const { return HashSuffix.has_value(); }
};

} // namespace proteus

#endif
