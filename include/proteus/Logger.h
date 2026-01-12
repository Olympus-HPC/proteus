#ifndef PROTEUS_LOGGER_H
#define PROTEUS_LOGGER_H

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

namespace proteus {

class Logger {
public:
  static llvm::raw_ostream &logs(const std::string &Name) {
    static Logger SingletonLogger{Name};
    SingletonLogger.OutStream << "[" << Name << "] ";
    return SingletonLogger.OutStream;
  }

  static llvm::raw_ostream &outs(const std::string &Name) {
    llvm::outs() << "[" << Name << "] ";
    return llvm::outs();
  }

  static void trace(llvm::StringRef Msg) { std::cout << Msg.str(); }

  template <typename T>
  static void logfile(const std::string &Filename, T &&Data) {
    std::error_code EC;
    llvm::raw_fd_ostream Out(std::string(LogDir) + "/" +
                                 std::to_string(getpid()) + "." + Filename,
                             EC);
    if (EC)
      throw std::runtime_error("Error opening file: " + EC.message());
    Out << Data;
    Out.close();
  }

private:
  static constexpr char LogDir[] = ".proteus-logs";
  bool DirExists;
  std::error_code EC;
  llvm::raw_fd_ostream OutStream;

  Logger(const std::string &Name)
      : DirExists(std::filesystem::create_directory(LogDir)),
        OutStream(llvm::raw_fd_ostream{std::string(LogDir) + "/" + Name + "." +
                                           std::to_string(getpid()) + ".log",
                                       EC, llvm::sys::fs::OF_None}) {
    if (EC)
      throw std::runtime_error("Error opening file: " + EC.message());

    // Synchronize C++ streams with stdio for tracing (e.g., printf from GPU
    // kernels).
    std::ios::sync_with_stdio(true);
  }
};

} // namespace proteus

#endif
