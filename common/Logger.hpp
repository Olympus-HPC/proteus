#include <filesystem>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <stdexcept>
#include <system_error>

namespace proteus {

class Logger {
public:
  static llvm::raw_ostream &logs(const std::string &Name) {
    static Logger SingletonLogger{Name};
    return SingletonLogger.OutStream;
  }

private:
  const std::string LogDir = ".proteus-logs";
  bool DirExists;
  std::error_code EC;
  llvm::raw_fd_ostream OutStream;

  Logger(std::string Name)
      : DirExists(std::filesystem::create_directory(LogDir)),
        OutStream(llvm::raw_fd_ostream{LogDir + "/" + Name + ".log", EC,
                                       llvm::sys::fs::OF_None}) {
    if (EC)
      throw std::runtime_error("Error opening file: " + EC.message());
  }
};

} // namespace proteus