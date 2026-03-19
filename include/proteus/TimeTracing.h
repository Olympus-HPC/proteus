//===-- TimeTracing.h -- Time tracing helpers -------------------*- C++ -*-===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TIME_TRACING_H
#define PROTEUS_TIME_TRACING_H

#include <chrono>
#include <cstdint>
#include <memory>
#include <string_view>

namespace proteus {

struct TimeTraceScopeWrapper;

class ScopedTimeTrace {
public:
  explicit ScopedTimeTrace(std::string_view Name);
  ~ScopedTimeTrace();

  ScopedTimeTrace(ScopedTimeTrace &&) = default;
  ScopedTimeTrace &operator=(ScopedTimeTrace &&) = default;

private:
  std::unique_ptr<TimeTraceScopeWrapper> Pimpl;
};

class Timer {
  using Clock = std::chrono::steady_clock;

public:
  explicit Timer(bool Enabled = true);

  uint64_t elapsed();

  template <class Duration> uint64_t elapsedAs() const {
    static_assert(
        !std::chrono::treat_as_floating_point_v<typename Duration::rep>,
        "Timer::elapsedAs only supports integral duration reps");
    if (!Enabled)
      return 0;
    return std::chrono::duration_cast<Duration>(Clock::now() - Start).count();
  }

  void reset();

private:
  bool Enabled = true;
  Clock::time_point Start;
};

// Use TIMESCOPE("Explicit::Label") for explicit phase names and
// TIMESCOPE(Class, Method) to emit a stable "Class::Method" label.
#define PROTEUS_TIMESCOPE_VAR(Line) STT_##Line
#define PROTEUS_TIMESCOPE_VAR_EXPAND(Line) PROTEUS_TIMESCOPE_VAR(Line)
#define PROTEUS_TIMESCOPE_1(Label)                                             \
  ::proteus::ScopedTimeTrace PROTEUS_TIMESCOPE_VAR_EXPAND(__LINE__)(Label);
#define PROTEUS_TIMESCOPE_2(Class, Method)                                     \
  PROTEUS_TIMESCOPE_1(#Class "::" #Method)
#define PROTEUS_GET_TIMESCOPE(_1, _2, NAME, ...) NAME
#define TIMESCOPE(...)                                                         \
  PROTEUS_GET_TIMESCOPE(__VA_ARGS__, PROTEUS_TIMESCOPE_2,                      \
                        PROTEUS_TIMESCOPE_1)(__VA_ARGS__)

} // namespace proteus

#endif
