#ifndef PROTEUS_UTILS_H
#define PROTEUS_UTILS_H

#include <string>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"

#include "TimeTracing.hpp"

#if ENABLE_DEBUG
#define DBG(x) x;
#else
#define DBG(x)
#endif

#define FATAL_ERROR(x)                                                         \
  report_fatal_error(llvm::Twine(std::string{} + __FILE__ + ":" +              \
                                 std::to_string(__LINE__) + " => " + x))

//  #define ENABLE_PERFMAP

#if ENABLE_HIP
#include "UtilsHIP.h"
#endif

#if ENABLE_CUDA
#include "UtilsCUDA.h"
#endif

#endif