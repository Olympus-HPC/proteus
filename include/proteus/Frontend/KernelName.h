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
// derived from its hash. The base name labels the kernel trace, the mangled
// name is the symbol in the compiled image.
class KernelName {
private:
  std::string Base;
  // Stores the mangled suffix, not the hash, so HashT can stay incomplete here.
  std::optional<std::string> HashSuffix;

public:
  // The name-only constructors are implicit on purpose.
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
