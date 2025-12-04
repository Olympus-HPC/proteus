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

#include "proteus/Config.hpp"

namespace proteus {

using namespace llvm;

struct TimeTracerRAII {
  TimeTracerRAII() { timeTraceProfilerInitialize(500 /* us */, "jit"); }

  ~TimeTracerRAII() {
    auto &OutputFile = Config::get().ProteusTimeTraceFile;
    if (auto E = timeTraceProfilerWrite(OutputFile.value_or(""), "-")) {
      handleAllErrors(std::move(E));
      return;
    }
    timeTraceProfilerCleanup();
  }
};

class Timer {
  using Clock = std::chrono::steady_clock;

public:
  Timer() {
    if (Config::get().ProteusEnableTimers)
      Start = Clock::now();
  }

  double elapsed() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                 Start)
        .count() / (static_cast<double>(1e6));
  }

  void reset() { Start = Clock::now(); }

private:
  Clock::time_point Start;
};

#define PROTEUS_TIMER_OUTPUT(x)                                                \
  if (Config::get().ProteusEnableTimers)                                       \
    x;

#if PROTEUS_ENABLE_TIME_TRACING
#define TIMESCOPE(x) TimeTraceScope TTS(x);
#else
#define TIMESCOPE(x)
#endif

} // namespace proteus

#endif
