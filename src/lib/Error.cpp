#include "proteus/Error.h"

// Include cstdint for LLVM headers missing it see
// https://bugzilla.redhat.com/show_bug.cgi?id=2338982.
#include <cstdint>
#include <llvm/ADT/Twine.h>

#include <sstream>
#include <string>
#include <unistd.h>

namespace proteus {

[[noreturn]] void reportFatalError(const llvm::Twine &Reason, const char *FILE,
                                   unsigned Line) {
  std::stringstream OS;
  OS << "[proteus][FATAL ERROR] " << FILE << ":" << Line << " => "
     << Reason.str() << "\n";

  std::string Message = OS.str();
  ::write(2, Message.data(), Message.size());

  std::abort();
}

[[noreturn]] void reportFatalError(const char *Reason, const char *FILE,
                                   unsigned Line) {
  reportFatalError(llvm::Twine(Reason), FILE, Line);
}

[[noreturn]] void reportFatalError(const std::string &Reason, const char *FILE,
                                   unsigned Line) {
  reportFatalError(llvm::Twine(Reason), FILE, Line);
}

[[noreturn]] void reportFatalError(const llvm::StringRef &Reason,
                                   const char *FILE, unsigned Line) {
  reportFatalError(llvm::Twine(Reason), FILE, Line);
}

} // namespace proteus
