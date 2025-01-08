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

#include "llvm/Support/TimeProfiler.h"
#include <chrono>
#include <iostream>

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

struct TimeTracerPrintout {
  TimeTracerPrintout(StringRef Tag)
      : Tag(Tag), Tstart(std::chrono::high_resolution_clock::now()) {}

  ~TimeTracerPrintout() {
    auto Tend = std::chrono::high_resolution_clock::now();
    auto Duration =
        std::chrono::duration_cast<std::chrono::microseconds>(Tend - Tstart)
            .count();
    std::cout << "[PROTEUS TRACE] " << Tag << " => " << Duration << " us\n";
  }

  std::chrono::high_resolution_clock::time_point Tstart;
  std::string Tag;
};

#if ENABLE_TIME_TRACING
#define TIMESCOPE(x) TimeTracerPrintout T(x);
#else
#define TIMESCOPE(x)
#endif

} // namespace proteus

#endif