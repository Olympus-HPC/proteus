#ifndef PROTEUS_LOGGER_HPP
#define PROTEUS_LOGGER_HPP

#include <filesystem>
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
    return SingletonLogger.OutStream;
  }

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
  }
};

} // namespace proteus

#endif