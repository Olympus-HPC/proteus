//===-- TimeTracing.hpp -- Time tracing helpers --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TIME_TRACING_HPP
#define PROTEUS_TIME_TRACING_HPP

#include <llvm/Support/TimeProfiler.h>

#include <chrono>

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

class Timer {
  using Clock = std::chrono::steady_clock;

public:
  Timer() { Start = Clock::now(); }

  uint64_t elapsed() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                                 Start)
        .count();
  }

  void reset() { Start = Clock::now(); }

private:
  Clock::time_point Start;
};

#if PROTEUS_ENABLE_TIME_TRACING
#define TIMESCOPE(x) TimeTraceScope TTS(x);
#else
#define TIMESCOPE(x)
#endif

} // namespace proteus

#endif