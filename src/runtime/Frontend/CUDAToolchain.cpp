#include "proteus/impl/Frontend/CUDAToolchain.h"

#include "proteus/Error.h"

#if PROTEUS_ENABLE_CUDA

#include "proteus/impl/Config.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>

#include <cctype>
#include <mutex>
#include <optional>
#include <utility>

namespace proteus {

namespace {

std::optional<std::string> getNonEmptyEnvVar(llvm::StringRef VarName) {
  std::string Name = VarName.str();
  auto Value = getEnvOrDefaultString(Name.c_str());
  if (!Value || Value->empty())
    return std::nullopt;
  return Value;
}

std::string canonicalizeExistingPath(llvm::StringRef Path) {
  llvm::SmallString<256> RealPath;
  if (!llvm::sys::fs::real_path(Path, RealPath))
    return RealPath.str().str();
  return Path.str();
}

std::string requireExistingDirectory(llvm::StringRef Path,
                                     llvm::StringRef Context) {
  if (!llvm::sys::fs::is_directory(Path))
    reportFatalError(Context.str() + ": expected directory, got " + Path.str());
  return canonicalizeExistingPath(Path);
}

std::string requireExistingFile(llvm::StringRef Path, llvm::StringRef Context) {
  if (!llvm::sys::fs::is_regular_file(Path))
    reportFatalError(Context.str() + ": expected file, got " + Path.str());
  return canonicalizeExistingPath(Path);
}

std::optional<std::pair<std::string, std::string>>
resolveRootFromEnv(llvm::StringRef VarName) {
  auto Value = getNonEmptyEnvVar(VarName);
  if (!Value)
    return std::nullopt;

  return std::pair<std::string, std::string>{
      requireExistingDirectory(*Value, VarName), VarName.str() + "=" + *Value};
}

std::string resolveLibDeviceFromRoot(llvm::StringRef Root) {
  llvm::SmallString<256> Candidate(Root);
  llvm::sys::path::append(Candidate, "nvvm", "libdevice", "libdevice.10.bc");
  return requireExistingFile(Candidate, "CUDA libdevice");
}

std::string resolveRuntimeLibDirFromRoot(llvm::StringRef Root) {
  // Host-CUDA JIT paths still need libcudart for registration symbols such as
  // __cudaRegisterFatBinary, so resolve a runtime library directory alongside
  // the toolkit root instead.
  for (llvm::StringRef RelativeDir : {"lib64", "lib"}) {
    llvm::SmallString<256> Candidate(Root);
    llvm::sys::path::append(Candidate, RelativeDir);
    if (llvm::sys::fs::is_directory(Candidate))
      return canonicalizeExistingPath(Candidate);
  }

  reportFatalError("Could not locate a CUDA runtime library directory under " +
                   Root.str() + ". Expected lib64 or lib.");
}

std::string readTextFile(llvm::StringRef Path, llvm::StringRef Context) {
  auto BufferOrErr = llvm::MemoryBuffer::getFile(Path);
  if (!BufferOrErr || !BufferOrErr.get())
    reportFatalError(Context.str() + ": failed to read " + Path.str());
  return std::string(BufferOrErr.get()->getBuffer());
}

std::optional<std::string> extractVersionFromVersionTxt(llvm::StringRef Text) {
  constexpr llvm::StringRef Prefix = "CUDA Version";
  std::size_t PrefixPos = Text.find(Prefix);
  if (PrefixPos == llvm::StringRef::npos)
    return std::nullopt;

  std::size_t Cursor = PrefixPos + Prefix.size();
  while (Cursor < Text.size() &&
         std::isspace(static_cast<unsigned char>(Text[Cursor])))
    ++Cursor;

  std::size_t End = Cursor;
  while (End < Text.size() &&
         (std::isdigit(static_cast<unsigned char>(Text[End])) ||
          Text[End] == '.')) {
    ++End;
  }

  if (End == Cursor)
    return std::nullopt;
  return Text.substr(Cursor, End - Cursor).str();
}

std::string readCUDAToolkitVersionFromJson(llvm::StringRef Path) {
  auto BufferOrErr = llvm::MemoryBuffer::getFile(Path);
  if (!BufferOrErr || !BufferOrErr.get())
    reportFatalError("CUDA version.json: failed to read " + Path.str());

  llvm::json::Value JsonValue =
      llvm::cantFail(llvm::json::parse(BufferOrErr.get()->getBuffer()),
                     "Cannot parse CUDA version.json");
  auto *RootObject = JsonValue.getAsObject();
  if (!RootObject)
    reportFatalError("CUDA version.json: top-level JSON is not an object in " +
                     Path.str());

  if (auto *CudaObject = RootObject->getObject("cuda")) {
    if (auto Version = CudaObject->getString("version"))
      return Version->str();
  }

  if (auto Version = RootObject->getString("version"))
    return Version->str();

  reportFatalError("CUDA version.json: missing version field in " + Path.str());
}

std::string readCUDAToolkitVersionFromRoot(llvm::StringRef Root) {
  llvm::SmallString<256> VersionJson(Root);
  llvm::sys::path::append(VersionJson, "version.json");
  if (llvm::sys::fs::is_regular_file(VersionJson))
    return readCUDAToolkitVersionFromJson(VersionJson);

  llvm::SmallString<256> VersionTxt(Root);
  llvm::sys::path::append(VersionTxt, "version.txt");
  if (llvm::sys::fs::is_regular_file(VersionTxt)) {
    std::string Text = readTextFile(VersionTxt, "CUDA version.txt");
    if (auto Version = extractVersionFromVersionTxt(Text))
      return *Version;
    reportFatalError("CUDA version.txt: could not parse version from " +
                     VersionTxt.str().str());
  }

  reportFatalError("Could not locate CUDA toolkit version metadata under " +
                   Root.str() + ". Expected version.json or version.txt.");
}

std::string extractMajorVersion(llvm::StringRef Version,
                                llvm::StringRef Context) {
  std::size_t End = 0;
  while (End < Version.size() &&
         std::isdigit(static_cast<unsigned char>(Version[End]))) {
    ++End;
  }

  if (End == 0) {
    reportFatalError(Context.str() + ": could not parse major version from " +
                     Version.str());
  }

  return Version.substr(0, End).str();
}

void validateCUDAToolkitVersion(llvm::StringRef Root, llvm::StringRef Origin) {
  // Keep the runtime-selected toolkit in the same major release family as the
  // one used to build Proteus so libdevice and host-side CUDA components do not
  // silently drift apart.
  constexpr llvm::StringRef BuildToolkitVersion = PROTEUS_CUDA_TOOLKIT_VERSION;
  std::string RuntimeToolkitVersion = readCUDAToolkitVersionFromRoot(Root);
  std::string BuildMajorVersion =
      extractMajorVersion(BuildToolkitVersion, "Build CUDA toolkit version");
  std::string RuntimeMajorVersion = extractMajorVersion(
      RuntimeToolkitVersion, "Runtime CUDA toolkit version");
  if (RuntimeMajorVersion != BuildMajorVersion) {
    reportFatalError("CUDA toolkit major version mismatch: build used " +
                     BuildToolkitVersion.str() + ", but " + Origin.str() +
                     " resolved to toolkit version " + RuntimeToolkitVersion +
                     " at " + Root.str());
  }
}

ResolvedCUDAToolchain resolveCUDAToolchainImpl() {
  std::optional<std::string> ExplicitLibDevice;
  std::optional<std::string> LibDeviceOrigin;
  if (auto Value = getNonEmptyEnvVar("PROTEUS_CUDA_LIBDEVICE_PATH")) {
    ExplicitLibDevice =
        requireExistingFile(*Value, "PROTEUS_CUDA_LIBDEVICE_PATH");
    LibDeviceOrigin = "PROTEUS_CUDA_LIBDEVICE_PATH=" + *Value;
  }

  std::optional<std::pair<std::string, std::string>> Root;
  for (llvm::StringRef Var : {"PROTEUS_CUDA_HOME", "CUDA_HOME", "CUDA_PATH"}) {
    Root = resolveRootFromEnv(Var);
    if (Root)
      break;
  }

  if (Root)
    validateCUDAToolkitVersion(Root->first, Root->second);

  if (ExplicitLibDevice) {
    // An explicit libdevice override can be used without a toolkit root for
    // LLVMIR-style flows, but when a root is available we still resolve the
    // runtime libdir for host-CUDA shared-library compilation.
    return {Root ? Root->first : "", *ExplicitLibDevice,
            Root ? resolveRuntimeLibDirFromRoot(Root->first) : "",
            Root ? Root->second + "; " + *LibDeviceOrigin : *LibDeviceOrigin};
  }

  if (!Root) {
    reportFatalError("Failed to resolve the CUDA toolkit root. Set "
                     "PROTEUS_CUDA_HOME, CUDA_HOME, or CUDA_PATH.");
  }

  return {Root->first, resolveLibDeviceFromRoot(Root->first),
          resolveRuntimeLibDirFromRoot(Root->first), Root->second};
}

} // namespace

const ResolvedCUDAToolchain &resolveCUDAToolchain() {
  static std::once_flag CacheOnce;
  static std::optional<ResolvedCUDAToolchain> Cache;

  std::call_once(CacheOnce, []() { Cache = resolveCUDAToolchainImpl(); });
  return *Cache;
}

} // namespace proteus

#endif

#if !PROTEUS_ENABLE_CUDA

namespace proteus {

const ResolvedCUDAToolchain &resolveCUDAToolchain() {
  reportFatalError("CUDA toolchain requested in a non-CUDA build");
  static const ResolvedCUDAToolchain Unreachable{};
  return Unreachable;
}

} // namespace proteus

#endif
