//===-- TimeTracingInit.h -- Time tracing initialization --------*- C++ -*-===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TIME_TRACING_INIT_H
#define PROTEUS_TIME_TRACING_INIT_H

#include <string>

namespace proteus {

class TimeTracerRAII {
public:
  TimeTracerRAII(bool Enable, std::string OutputFile, int GranularityUs);
  ~TimeTracerRAII();

  TimeTracerRAII(const TimeTracerRAII &) = delete;
  TimeTracerRAII &operator=(const TimeTracerRAII &) = delete;

private:
  bool Enabled = false;
  std::string OutputFile;
};

class TimeTraceThreadRAII {
public:
  TimeTraceThreadRAII();
  ~TimeTraceThreadRAII();

  TimeTraceThreadRAII(const TimeTraceThreadRAII &) = delete;
  TimeTraceThreadRAII &operator=(const TimeTraceThreadRAII &) = delete;

private:
  bool Enabled = false;
};

} // namespace proteus

#endif
