#include "LLVMModuleBindings.h"

#include "proteus/impl/LLVMModule.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace proteus_python {
namespace {

using proteus::LLVMCodeGenerationConfig;
using proteus::LLVMModule;

struct ConfigView {};

std::string normalizeMethod(std::string Method) {
  std::transform(Method.begin(), Method.end(), Method.begin(),
                 [](unsigned char C) { return std::tolower(C); });
  if (Method != "rtc" && Method != "serial" && Method != "parallel")
    throw py::value_error("method must be 'rtc', 'serial', or 'parallel'");
  return Method;
}

void validateOptLevel(const std::string &OptLevel) {
  if (OptLevel.size() != 2 || OptLevel[0] != 'O' ||
      std::string("0123sz").find(OptLevel[1]) == std::string::npos)
    throw py::value_error("opt_level must be O0, O1, O2, O3, Os, or Oz");
}

void validateConfig(const LLVMCodeGenerationConfig &Config) {
  normalizeMethod(Config.Method);
  validateOptLevel(Config.OptLevel);
  if (Config.CodegenOptLevel > 3)
    throw py::value_error("codegen_opt_level must be between 0 and 3");
  if (Config.TunedMaxThreads && *Config.TunedMaxThreads <= 0)
    throw py::value_error("tuned_max_threads must be positive");
  if (Config.MinBlocksPerSM < 0)
    throw py::value_error("min_blocks_per_sm cannot be negative");
}

LLVMCodeGenerationConfig replaceConfig(const LLVMCodeGenerationConfig &Original,
                                       const py::kwargs &Kwargs) {
  LLVMCodeGenerationConfig Result = Original;
  for (auto Item : Kwargs) {
    const std::string Key = py::cast<std::string>(Item.first);
    const py::handle Value = Item.second;
    if (Key == "pipeline") {
      if (Value.is_none())
        Result.Pipeline.reset();
      else
        Result.Pipeline = py::cast<std::string>(Value);
    } else if (Key == "method") {
      Result.Method = normalizeMethod(py::cast<std::string>(Value));
    } else if (Key == "opt_level") {
      Result.OptLevel = py::cast<std::string>(Value);
    } else if (Key == "codegen_opt_level") {
      Result.CodegenOptLevel = py::cast<unsigned>(Value);
    } else if (Key == "specialize_arguments") {
      Result.SpecializeArguments = py::cast<bool>(Value);
    } else if (Key == "set_launch_bounds") {
      Result.SpecializeLaunchBounds = py::cast<bool>(Value);
    } else if (Key == "specialize_dimensions") {
      Result.SpecializeDimensions = py::cast<bool>(Value);
    } else if (Key == "specialize_dimension_ranges") {
      Result.SpecializeDimensionRanges = py::cast<bool>(Value);
    } else if (Key == "tuned_max_threads") {
      if (Value.is_none())
        Result.TunedMaxThreads.reset();
      else
        Result.TunedMaxThreads = py::cast<int>(Value);
    } else if (Key == "min_blocks_per_sm") {
      Result.MinBlocksPerSM = py::cast<int>(Value);
    } else {
      throw py::type_error("unexpected configuration field '" + Key + "'");
    }
  }
  validateConfig(Result);
  return Result;
}

LLVMModule::Dimensions parseDimensions(const py::sequence &Value,
                                       const char *Name) {
  if (py::len(Value) != 3)
    throw py::value_error(std::string(Name) +
                          " must contain exactly three dimensions");

  LLVMModule::Dimensions Result;
  for (py::ssize_t I = 0; I < 3; ++I) {
    const long long Dimension = py::cast<long long>(Value[I]);
    if (Dimension <= 0 || static_cast<unsigned long long>(Dimension) >
                              std::numeric_limits<uint32_t>::max())
      throw py::value_error(std::string(Name) +
                            " dimensions must be positive uint32 values");
    Result[I] = static_cast<uint32_t>(Dimension);
  }
  return Result;
}

[[noreturn]] void
raiseNotImplemented(const proteus::LLVMBackendUnavailableError &Error) {
  PyErr_SetString(PyExc_NotImplementedError, Error.what());
  throw py::error_already_set();
}

template <typename Callable>
decltype(auto) translateUnsupported(Callable &&Fn) {
  try {
    return Fn();
  } catch (const proteus::LLVMBackendUnavailableError &Error) {
    raiseNotImplemented(Error);
  }
}

std::string configRepr(const LLVMCodeGenerationConfig &Config) {
  return "<proteus.CodeGenerationConfig method='" + Config.Method +
         "' opt_level='" + Config.OptLevel +
         "' codegen_opt_level=" + std::to_string(Config.CodegenOptLevel) + ">";
}

} // namespace

void bindLLVMModule(py::module_ &M) {
  py::class_<LLVMCodeGenerationConfig>(M, "CodeGenerationConfig")
      .def_property_readonly(
          "pipeline",
          [](const LLVMCodeGenerationConfig &Config) -> py::object {
            if (!Config.Pipeline)
              return py::none();
            return py::str(*Config.Pipeline);
          })
      .def_property_readonly(
          "method",
          [](const LLVMCodeGenerationConfig &Config) { return Config.Method; })
      .def_property_readonly("opt_level",
                             [](const LLVMCodeGenerationConfig &Config) {
                               return Config.OptLevel;
                             })
      .def_property_readonly("codegen_opt_level",
                             [](const LLVMCodeGenerationConfig &Config) {
                               return Config.CodegenOptLevel;
                             })
      .def_property_readonly("specialize_arguments",
                             [](const LLVMCodeGenerationConfig &Config) {
                               return Config.SpecializeArguments;
                             })
      .def_property_readonly("set_launch_bounds",
                             [](const LLVMCodeGenerationConfig &Config) {
                               return Config.SpecializeLaunchBounds;
                             })
      .def_property_readonly("specialize_dimensions",
                             [](const LLVMCodeGenerationConfig &Config) {
                               return Config.SpecializeDimensions;
                             })
      .def_property_readonly("specialize_dimension_ranges",
                             [](const LLVMCodeGenerationConfig &Config) {
                               return Config.SpecializeDimensionRanges;
                             })
      .def_property_readonly(
          "tuned_max_threads",
          [](const LLVMCodeGenerationConfig &Config) -> py::object {
            if (!Config.TunedMaxThreads)
              return py::none();
            return py::int_(*Config.TunedMaxThreads);
          })
      .def_property_readonly("min_blocks_per_sm",
                             [](const LLVMCodeGenerationConfig &Config) {
                               return Config.MinBlocksPerSM;
                             })
      .def("replace", &replaceConfig)
      .def("__repr__", &configRepr);

  py::class_<ConfigView>(M, "Config")
      .def_static("current", [] { return ConfigView{}; })
      .def(
          "codegen",
          [](const ConfigView &, const py::object &KernelName) {
            if (KernelName.is_none())
              return proteus::getLLVMCodeGenerationConfig();
            return proteus::getLLVMCodeGenerationConfig(
                py::cast<std::string>(KernelName));
          },
          py::arg("kernel_name") = py::none());

  py::module_ LLVM = M.def_submodule(
      "llvm", "Owned staged LLVM modules and replay transformations");
  py::register_exception<proteus::LLVMSymbolNotFoundError>(
      LLVM, "SymbolNotFoundError", PyExc_KeyError);

  py::class_<LLVMModule>(LLVM, "Module")
      .def_static(
          "from_bitcode",
          [](const py::bytes &Data) {
            const std::string Bitcode = Data;
            py::gil_scoped_release Release;
            return LLVMModule::fromBitcode(Bitcode);
          },
          py::arg("data"))
      .def_static(
          "from_ir",
          [](const std::string &IR, const std::string &Name) {
            py::gil_scoped_release Release;
            return LLVMModule::fromIR(IR, Name);
          },
          py::arg("text"), py::arg("name") = "<string>")
      .def_static(
          "link",
          [](const py::iterable &Modules) {
            std::vector<const LLVMModule *> Inputs;
            for (const py::handle Item : Modules)
              Inputs.push_back(py::cast<const LLVMModule *>(Item));
            py::gil_scoped_release Release;
            return LLVMModule::link(Inputs);
          },
          py::arg("modules"))
      .def("clone",
           [](const LLVMModule &Module) {
             py::gil_scoped_release Release;
             return Module.clone();
           })
      .def("to_bitcode",
           [](const LLVMModule &Module) {
             std::string Bitcode;
             {
               py::gil_scoped_release Release;
               Bitcode = Module.toBitcode();
             }
             return py::bytes(Bitcode.data(), Bitcode.size());
           })
      .def("to_ir",
           [](const LLVMModule &Module) {
             py::gil_scoped_release Release;
             return Module.toIR();
           })
      .def("verify",
           [](const LLVMModule &Module) {
             py::gil_scoped_release Release;
             Module.verify();
           })
      .def(
          "prune",
          [](LLVMModule &Module,
             bool UnsetExternallyInitialized) -> LLVMModule & {
            py::gil_scoped_release Release;
            return Module.prune(UnsetExternallyInitialized);
          },
          py::arg("unset_externally_initialized") = true,
          py::return_value_policy::reference_internal)
      .def(
          "internalize",
          [](LLVMModule &Module,
             const std::vector<std::string> &Preserve) -> LLVMModule & {
            py::gil_scoped_release Release;
            return Module.internalize(Preserve);
          },
          py::kw_only(), py::arg("preserve"),
          py::return_value_policy::reference_internal)
      .def(
          "specialize_arguments",
          [](LLVMModule &Module, const std::string &KernelName,
             const py::buffer &Arguments,
             const std::vector<std::size_t> &Indexes) -> LLVMModule & {
            py::buffer_info Info = Arguments.request();
            if (Info.ndim != 1)
              throw py::value_error(
                  "arguments must be a one-dimensional buffer");
            if (Info.itemsize != static_cast<py::ssize_t>(sizeof(void *)))
              throw py::value_error(
                  "arguments buffer entries must be native pointer-sized");
            if (Info.strides[0] != Info.itemsize)
              throw py::value_error("arguments buffer must be C-contiguous");
            return translateUnsupported([&]() -> LLVMModule & {
              return Module.specializeArguments(
                  KernelName, static_cast<void *const *>(Info.ptr), Info.size,
                  Indexes);
            });
          },
          py::arg("kernel_name"), py::kw_only(), py::arg("arguments"),
          py::arg("indexes"), py::return_value_policy::reference_internal)
      .def(
          "specialize_launch_dimensions",
          [](LLVMModule &Module, const py::sequence &Grid,
             const py::sequence &Block) -> LLVMModule & {
            const auto GridDims = parseDimensions(Grid, "grid");
            const auto BlockDims = parseDimensions(Block, "block");
            return translateUnsupported([&]() -> LLVMModule & {
              return Module.specializeLaunchDimensions(GridDims, BlockDims);
            });
          },
          py::kw_only(), py::arg("grid"), py::arg("block"),
          py::return_value_policy::reference_internal)
      .def(
          "assume_launch_dimension_ranges",
          [](LLVMModule &Module, const py::sequence &Grid,
             const py::sequence &Block) -> LLVMModule & {
            const auto GridDims = parseDimensions(Grid, "grid");
            const auto BlockDims = parseDimensions(Block, "block");
            return translateUnsupported([&]() -> LLVMModule & {
              return Module.assumeLaunchDimensionRanges(GridDims, BlockDims);
            });
          },
          py::kw_only(), py::arg("grid"), py::arg("block"),
          py::return_value_policy::reference_internal)
      .def(
          "set_launch_bounds",
          [](LLVMModule &Module, const std::string &KernelName,
             unsigned MaxThreadsPerBlock,
             unsigned MinBlocksPerSM) -> LLVMModule & {
            return translateUnsupported([&]() -> LLVMModule & {
              return Module.setLaunchBounds(KernelName, MaxThreadsPerBlock,
                                            MinBlocksPerSM);
            });
          },
          py::arg("kernel_name"), py::kw_only(),
          py::arg("max_threads_per_block"), py::arg("min_blocks_per_sm") = 0,
          py::return_value_policy::reference_internal)
      .def(
          "optimize",
          [](LLVMModule &Module, const std::string &DeviceArch,
             const LLVMCodeGenerationConfig &Config) -> LLVMModule & {
            py::gil_scoped_release Release;
            return Module.optimize(DeviceArch, Config);
          },
          py::arg("device_arch"), py::kw_only(), py::arg("config"),
          py::return_value_policy::reference_internal)
      .def(
          "emit_object",
          [](const LLVMModule &Module, const std::string &DeviceArch,
             const LLVMCodeGenerationConfig &Config) {
            std::string Object;
            try {
              py::gil_scoped_release Release;
              Object = Module.emitObject(DeviceArch, Config);
            } catch (const proteus::LLVMBackendUnavailableError &Error) {
              raiseNotImplemented(Error);
            }
            return py::bytes(Object.data(), Object.size());
          },
          py::arg("device_arch"), py::kw_only(), py::arg("config"));
}

} // namespace proteus_python
