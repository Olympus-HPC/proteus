#ifndef PROTEUS_ERROR_H
#define PROTEUS_ERROR_H

#include <cstdint>
#include <llvm/ADT/Twine.h>

#define PROTEUS_FATAL_ERROR(x)                                                 \
  report_fatal_error(llvm::Twine(std::string{} + __FILE__ + ":" +              \
                                 std::to_string(__LINE__) + " => " + x))

#endif
