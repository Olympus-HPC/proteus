#ifndef PROTEUS_TIME_TRACING_HPP
#define PROTEUS_TIME_TRACING_HPP

#include "llvm/Support/TimeProfiler.h"

namespace proteus {

using namespace llvm;

struct TimeTracerRAII {
  TimeTracerRAII() { timeTraceProfilerInitialize(500 /* us */, "jit"); }

  ~TimeTracerRAII() {
    if (auto E = timeTraceProfilerWrite("", "-")) {
      handleAllErrors(std::move(E));
      return;
    }
    timeTraceProfilerCleanup();
  }
};

#if ENABLE_TIME_TRACING
TimeTracerRAII TimeTracer;
#define TIMESCOPE(x) TimeTraceScope T(x);
#else
#define TIMESCOPE(x)
#endif

} // namespace proteus

#endif