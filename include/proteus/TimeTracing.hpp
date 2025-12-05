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

#include <chrono>
#include <optional>

#include <llvm/Support/TimeProfiler.h>

#include "proteus/Config.hpp"

namespace proteus {

using namespace llvm;

using TimeTraceOptional = std::optional<TimeTraceScope>;

struct TimeTracerRAII {
  TimeTracerRAII() {
    if (Config::get().ProteusEnableTimeTrace) {
      timeTraceProfilerInitialize(500 /* us */, "proteus");
    }
  }

  ~TimeTracerRAII() {
    if (Config::get().ProteusEnableTimeTrace) {
      auto &OutputFile = Config::get().ProteusTimeTraceFile;
      if (auto E = timeTraceProfilerWrite(OutputFile, "-")) {
        handleAllErrors(std::move(E));
        return;
      }
      timeTraceProfilerCleanup();
    }
  }
};

class Timer {
  using Clock = std::chrono::steady_clock;

public:
  Timer() {
    if (Config::get().ProteusEnableTimers)
      Start = Clock::now();
  }

  uint64_t elapsed() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                                 Start)
        .count();
  }

  void reset() { Start = Clock::now(); }

private:
  Clock::time_point Start;
};

#define PROTEUS_TIMER_OUTPUT(x)                                                \
  if (Config::get().ProteusEnableTimers)                                       \
    x;

#define TIMESCOPE(x)                                                           \
  TimeTraceOptional TTS;                                                       \
  if (Config::get().ProteusEnableTimeTrace) {                                  \
    TTS.emplace(x);                                                            \
  }

} // namespace proteus

#endif
