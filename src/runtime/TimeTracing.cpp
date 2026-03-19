#include "proteus/TimeTracing.h"
#include "proteus/impl/TimeTracingInit.h"

#include <llvm/Support/Error.h>
#include <llvm/Support/TimeProfiler.h>

namespace proteus {

namespace {
unsigned ProcessTimeTraceGranularityUs = 0;
constexpr llvm::StringLiteral TimeTraceProcessName = "proteus";
} // namespace

struct TimeTraceScopeWrapper {
  llvm::TimeTraceScope Scope;

  explicit TimeTraceScopeWrapper(std::string_view Name) : Scope(Name) {}
};

ScopedTimeTrace::ScopedTimeTrace(std::string_view Name) {
  if (llvm::timeTraceProfilerEnabled())
    Pimpl = std::make_unique<TimeTraceScopeWrapper>(Name);
}

ScopedTimeTrace::~ScopedTimeTrace() = default;

TimeTracerRAII::TimeTracerRAII(bool Enable, std::string OutputFile,
                               int GranularityUs)
    : Enabled(Enable), OutputFile(std::move(OutputFile)) {
  if (!Enabled)
    return;

  ProcessTimeTraceGranularityUs = GranularityUs;
  llvm::timeTraceProfilerInitialize(ProcessTimeTraceGranularityUs,
                                    TimeTraceProcessName);
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

TimeTraceThreadRAII::TimeTraceThreadRAII() {
  Enabled = llvm::timeTraceProfilerEnabled();
  if (Enabled)
    llvm::timeTraceProfilerInitialize(ProcessTimeTraceGranularityUs,
                                      TimeTraceProcessName);
}

TimeTraceThreadRAII::~TimeTraceThreadRAII() {
  if (Enabled)
    llvm::timeTraceProfilerFinishThread();
}

Timer::Timer(bool Enabled) : Enabled(Enabled) {
  if (this->Enabled)
    Start = Clock::now();
}

uint64_t Timer::elapsed() { return elapsedAs<std::chrono::milliseconds>(); }

void Timer::reset() {
  if (!Enabled)
    return;
  Start = Clock::now();
}

} // namespace proteus
