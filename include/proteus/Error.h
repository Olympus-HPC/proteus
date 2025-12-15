#ifndef PROTEUS_ERROR_H
#define PROTEUS_ERROR_H

#include <string>

namespace llvm {
class StringRef;
class Twine;
} // namespace llvm

namespace proteus {

[[noreturn]] void reportFatalError(const char *Reason,
                                   const char *FILE = __builtin_FILE(),
                                   unsigned Line = __builtin_LINE());

[[noreturn]] void reportFatalError(const std::string &Reason,
                                   const char *FILE = __builtin_FILE(),
                                   unsigned Line = __builtin_LINE());

[[noreturn]] void reportFatalError(const llvm::StringRef &Reason,
                                   const char *FILE = __builtin_FILE(),
                                   unsigned Line = __builtin_LINE());

[[noreturn]] void reportFatalError(const llvm::Twine &Reason,
                                   const char *FILE = __builtin_FILE(),
                                   unsigned Line = __builtin_LINE());

} // namespace proteus

#endif
