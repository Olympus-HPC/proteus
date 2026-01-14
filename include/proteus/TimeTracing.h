//===-- TimeTracing.h -- Time tracing helpers --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TIME_TRACING_H
#define PROTEUS_TIME_TRACING_H

#include <chrono>
#include <memory>

namespace proteus {

// RAII wrapper that handles the optional logic internally
struct TimeTraceScopeWrapper;
class ScopedTimeTrace {
public:
  explicit ScopedTimeTrace(const std::string &Name);
  ~ScopedTimeTrace();

  // Support only move-only construction.
  ScopedTimeTrace(ScopedTimeTrace &&) = default;
  ScopedTimeTrace &operator=(ScopedTimeTrace &&) = default;

private:
  std::unique_ptr<TimeTraceScopeWrapper> Pimpl;
};

struct TimeTracerRAII {
  TimeTracerRAII();

  ~TimeTracerRAII();
};

class Timer {
  using Clock = std::chrono::steady_clock;

public:
  Timer();

  uint64_t elapsed();

  void reset();

private:
  Clock::time_point Start;
};

#define PROTEUS_TIMER_OUTPUT(x)                                                \
  if (Config::get().ProteusEnableTimers)                                       \
    x;

// Macro now creates the wrapper, which is lightweight in the header.
#define TIMESCOPE(x) proteus::ScopedTimeTrace STT_##__LINE__(x);

} // namespace proteus

#endif
