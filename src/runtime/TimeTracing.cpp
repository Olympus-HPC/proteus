#include "proteus/TimeTracing.h"

#include "proteus/impl/TimeTracingInit.h"

#include <llvm/Support/Error.h>
#include <llvm/Support/TimeProfiler.h>

namespace proteus {

struct TimeTraceScopeWrapper {
  std::string NameStorage;
  llvm::TimeTraceScope Scope;

  explicit TimeTraceScopeWrapper(std::string_view Name)
      : NameStorage(Name), Scope(NameStorage) {}
};

ScopedTimeTrace::ScopedTimeTrace(std::string_view Name) {
  if (llvm::timeTraceProfilerEnabled())
    Pimpl = std::make_unique<TimeTraceScopeWrapper>(Name);
}

ScopedTimeTrace::~ScopedTimeTrace() = default;

TimeTracerRAII::TimeTracerRAII(bool Enable, std::string OutputFile)
    : Enabled(Enable), OutputFile(std::move(OutputFile)) {
  if (Enabled)
    llvm::timeTraceProfilerInitialize(500 /* us */, "proteus");
}

TimeTracerRAII::~TimeTracerRAII() {
  if (!Enabled)
    return;

  if (auto E = llvm::timeTraceProfilerWrite(OutputFile, "-")) {
    llvm::handleAllErrors(std::move(E));
    return;
  }
  llvm::timeTraceProfilerCleanup();
}

Timer::Timer(bool Enabled) : Enabled(Enabled) {
  if (this->Enabled)
    Start = Clock::now();
}

uint64_t Timer::elapsed() {
  return elapsedAs<std::chrono::milliseconds>();
}

void Timer::reset() {
  if (!Enabled)
    return;
  Start = Clock::now();
}

} // namespace proteus
