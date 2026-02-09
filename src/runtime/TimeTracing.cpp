
#include "proteus/impl/TimeTracing.h"
#include "proteus/impl/Config.h"

#include <llvm/Support/TimeProfiler.h>

#include <chrono>

namespace proteus {

using namespace llvm;

// Define the wrapper struct to hold the actual LLVM type
struct TimeTraceScopeWrapper {
  TimeTraceScope Scope;
  TimeTraceScopeWrapper(StringRef Name) : Scope(Name) {}
};

ScopedTimeTrace::ScopedTimeTrace(const std::string &Name) {
  if (Config::get().ProteusEnableTimeTrace) {
    Pimpl = std::make_unique<TimeTraceScopeWrapper>(Name);
  }
}

ScopedTimeTrace::~ScopedTimeTrace() = default;

TimeTracerRAII::TimeTracerRAII() {
  if (Config::get().ProteusEnableTimeTrace) {
    timeTraceProfilerInitialize(500 /* us */, "proteus");
  }
}

TimeTracerRAII::~TimeTracerRAII() {
  if (Config::get().ProteusEnableTimeTrace) {
    auto &OutputFile = Config::get().ProteusTimeTraceFile;
    if (auto E = timeTraceProfilerWrite(OutputFile, "-")) {
      handleAllErrors(std::move(E));
      return;
    }
    timeTraceProfilerCleanup();
  }
}

using Clock = std::chrono::steady_clock;

Timer::Timer() {
  if (Config::get().ProteusEnableTimers)
    Start = Clock::now();
}

uint64_t Timer::elapsed() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                               Start)
      .count();
}

void Timer::reset() { Start = Clock::now(); }

} // namespace proteus
