#include "PythonBindings.h"
#include "LLVMModuleBindings.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace proteus_python {
namespace {

struct ArgStorage {
  // Large enough for the scalar and pointer argument types exposed in Python.
  alignas(std::max_align_t) std::array<unsigned char, 16> Data{};

  template <typename T> void set(T Value) {
    static_assert(sizeof(T) <= 16, "argument storage is too small");
    new (Data.data()) T(Value);
  }

  void *ptr() { return Data.data(); }
};

std::string readSource(py::object Source) {
  // Read from a file when Source's string representation names one; otherwise,
  // treat the string itself as inline source text.
  py::object PathLike = Source;
  if (py::hasattr(Source, "__fspath__"))
    PathLike = Source.attr("__fspath__")();

  std::string Text = py::str(PathLike);
  std::ifstream Input(Text);
  if (!Input)
    return Text;

  std::ostringstream OS;
  OS << Input.rdbuf();
  return OS.str();
}

LaunchDims parseDims(py::object Obj, const char *Name) {
  // A scalar launch size is treated as the x dimension with y=z=1.
  if (py::isinstance<py::int_>(Obj)) {
    auto X = Obj.cast<unsigned>();
    return LaunchDims{X, 1, 1};
  }

  py::sequence Seq;
  try {
    Seq = Obj.cast<py::sequence>();
  } catch (const py::cast_error &) {
    throw py::type_error(std::string(Name) +
                         " must be an int or a tuple/list of 1 to 3 ints");
  }

  if (Seq.size() < 1 || Seq.size() > 3)
    throw py::type_error(std::string(Name) +
                         " must contain between 1 and 3 dimensions");

  unsigned X = Seq[0].cast<unsigned>();
  unsigned Y = Seq.size() > 1 ? Seq[1].cast<unsigned>() : 1;
  unsigned Z = Seq.size() > 2 ? Seq[2].cast<unsigned>() : 1;
  return LaunchDims{X, Y, Z};
}

uintptr_t ptrFromInterface(py::object Obj, const char *AttrName) {
  // Both array interface protocols expose the raw pointer in the first slot of
  // the `data` tuple.
  py::dict Interface = Obj.attr(AttrName).cast<py::dict>();
  py::tuple Data = Interface["data"].cast<py::tuple>();
  if (Data.size() < 1)
    throw py::type_error(std::string(AttrName) + " data tuple is empty");
  return Data[0].cast<uintptr_t>();
}

uintptr_t ptrFromHostObject(py::object Obj) {
  if (Obj.is_none())
    return 0;

  if (py::isinstance<py::int_>(Obj))
    return Obj.cast<uintptr_t>();

  if (py::hasattr(Obj, "data_ptr"))
    // Match tensor libraries that surface a callable `data_ptr()` helper.
    return Obj.attr("data_ptr")().cast<uintptr_t>();

  if (py::hasattr(Obj, "__array_interface__"))
    // Reuse host pointers from array-like objects such as NumPy arrays.
    return ptrFromInterface(Obj, "__array_interface__");

  py::module_ ctypes = py::module_::import("ctypes");
  try {
    // Pointer wrapper objects such as `ctypes.pointer(x)` need a cast to
    // expose the pointee address instead of the wrapper object's storage.
    py::object Ptr = ctypes.attr("cast")(Obj, ctypes.attr("c_void_p"));
    if (!Ptr.is_none()) {
      py::object Value = Ptr.attr("value");
      if (!Value.is_none())
        return Value.cast<uintptr_t>();
    }
  } catch (const py::error_already_set &) {
    PyErr_Clear();
  }
  try {
    // `ctypes.addressof()` covers explicit ctypes instances and buffers that do
    // not expose array-style protocols.
    return ctypes.attr("addressof")(Obj).cast<uintptr_t>();
  } catch (const py::error_already_set &) {
    PyErr_Clear();
  }

  throw py::type_error("pointer argument must be an int, None, data_ptr() "
                       "object, __array_interface__ object, or ctypes object");
}

uintptr_t ptrFromDeviceObject(py::object Obj) {
  if (Obj.is_none())
    return 0;

  if (py::isinstance<py::int_>(Obj))
    return Obj.cast<uintptr_t>();

  if (py::hasattr(Obj, "data_ptr"))
    // Match tensor libraries that surface a callable `data_ptr()` helper.
    return Obj.attr("data_ptr")().cast<uintptr_t>();

  if (py::hasattr(Obj, "__cuda_array_interface__"))
    // Reuse device pointers from GPU array-like objects exposing the CUDA
    // array interface protocol.
    return ptrFromInterface(Obj, "__cuda_array_interface__");

  throw py::type_error("pointer argument must be an int, None, data_ptr() "
                       "object, or __cuda_array_interface__ object");
}

ArgStorage convertArg(Type Ty, py::object Value, bool DevicePointers) {
  // Marshal Python objects into stable storage before taking argument
  // addresses.
  ArgStorage Storage;
  switch (Ty.Kind) {
  case PyType::Void:
    throw py::type_error(
        "proteus.void is only valid as a signature return type");
  case PyType::I8:
    Storage.set<int8_t>(Value.cast<int8_t>());
    break;
  case PyType::I32:
    Storage.set<int32_t>(Value.cast<int32_t>());
    break;
  case PyType::I64:
    Storage.set<int64_t>(Value.cast<int64_t>());
    break;
  case PyType::U32:
    Storage.set<uint32_t>(Value.cast<uint32_t>());
    break;
  case PyType::U64:
    Storage.set<uint64_t>(Value.cast<uint64_t>());
    break;
  case PyType::F32:
    Storage.set<float>(Value.cast<float>());
    break;
  case PyType::F64:
    Storage.set<double>(Value.cast<double>());
    break;
  case PyType::Ptr:
    Storage.set<uintptr_t>(DevicePointers ? ptrFromDeviceObject(Value)
                                          : ptrFromHostObject(Value));
    break;
  }
  return Storage;
}

class Module;
class Function;

std::string typeRepr(Type Ty) {
  switch (Ty.Kind) {
  case PyType::Void:
    return "proteus.void";
  case PyType::I8:
    return "proteus.i8";
  case PyType::I32:
    return "proteus.i32";
  case PyType::I64:
    return "proteus.i64";
  case PyType::U32:
    return "proteus.u32";
  case PyType::U64:
    return "proteus.u64";
  case PyType::F32:
    return "proteus.f32";
  case PyType::F64:
    return "proteus.f64";
  case PyType::Ptr:
    return "proteus.ptr";
  }
  throw py::type_error("unsupported Proteus type descriptor");
}

std::string signatureRepr(const Signature &Sig) {
  std::string Repr = typeRepr(Sig.RetType) + "(";
  for (std::size_t I = 0; I < Sig.ArgTypes.size(); ++I) {
    if (I != 0)
      Repr += ", ";
    Repr += typeRepr(Sig.ArgTypes[I]);
  }
  Repr += ")";
  return Repr;
}

void validateSignature(const Signature &Sig) {
  for (Type ArgType : Sig.ArgTypes)
    if (ArgType.Kind == PyType::Void)
      throw py::type_error(
          "proteus.void is only valid as a signature return type");
}

Signature makeSignature(Type RetType, py::args Args) {
  std::vector<Type> ArgTypes;
  ArgTypes.reserve(Args.size());
  for (py::handle Arg : Args) {
    if (!py::isinstance<Type>(Arg))
      throw py::type_error(
          "signature arguments must be Proteus type descriptors");
    Type ArgType = py::cast<Type>(Arg);
    if (ArgType.Kind == PyType::Void)
      throw py::type_error(
          "proteus.void is only valid as a signature return type");
    ArgTypes.push_back(ArgType);
  }
  return Signature{RetType, std::move(ArgTypes)};
}

py::object typeToCType(py::handle ctypes, Type Ty) {
  // Build the ctypes signature from the lightweight descriptors exported by
  // this module.
  switch (Ty.Kind) {
  case PyType::Void:
    return py::none();
  case PyType::I8:
    return ctypes.attr("c_int8");
  case PyType::I32:
    return ctypes.attr("c_int32");
  case PyType::I64:
    return ctypes.attr("c_int64");
  case PyType::U32:
    return ctypes.attr("c_uint32");
  case PyType::U64:
    return ctypes.attr("c_uint64");
  case PyType::F32:
    return ctypes.attr("c_float");
  case PyType::F64:
    return ctypes.attr("c_double");
  case PyType::Ptr:
    return ctypes.attr("c_void_p");
  }
  throw py::type_error("unsupported type");
}

class Kernel {
  std::shared_ptr<ModuleBase> Mod;
  void *KernelFunc = nullptr;
  Signature Sig;
  std::string Name;

public:
  Kernel(std::shared_ptr<ModuleBase> Mod, void *KernelFunc, Signature Sig,
         std::string Name)
      : Mod(std::move(Mod)), KernelFunc(KernelFunc), Sig(std::move(Sig)),
        Name(std::move(Name)) {}

  void launch(py::object Grid, py::object Block, py::sequence Args,
              uint64_t Shmem, py::object Stream) {
    if (Args.size() != Sig.ArgTypes.size())
      throw py::type_error("kernel argument count does not match signature");

    // Keep the converted values alive for the duration of the kernel launch.
    std::vector<ArgStorage> Storage;
    Storage.reserve(Sig.ArgTypes.size());
    for (std::size_t I = 0; I < Sig.ArgTypes.size(); ++I)
      Storage.push_back(convertArg(
          Sig.ArgTypes[I], py::reinterpret_borrow<py::object>(Args[I]), true));

    std::vector<void *> RawArgs;
    RawArgs.reserve(Storage.size());
    for (auto &Arg : Storage)
      // Proteus expects an argv-style array of pointers to each marshalled
      // argument value.
      RawArgs.push_back(Arg.ptr());

    void *StreamPtr = nullptr;
    if (!Stream.is_none())
      // Streams are passed through as opaque integer-backed handles.
      StreamPtr = reinterpret_cast<void *>(Stream.cast<uintptr_t>());

    auto Result = Mod->launch(KernelFunc, parseDims(Grid, "grid"),
                              parseDims(Block, "block"), RawArgs.data(), Shmem,
                              StreamPtr);
    if (static_cast<int>(Result) != 0)
      throw std::runtime_error("Proteus kernel launch failed with error code " +
                               std::to_string(static_cast<int>(Result)));
  }

  std::string repr() const {
    return "<proteus.Kernel name='" + Name +
           "' signature=" + signatureRepr(Sig) + ">";
  }
};

class Function {
  std::shared_ptr<ModuleBase> Mod;
  py::object Callable;
  std::string Name;
  Signature Sig;

public:
  Function(std::shared_ptr<ModuleBase> Mod, py::object Callable,
           std::string Name, Signature Sig)
      : Mod(std::move(Mod)), Callable(std::move(Callable)),
        Name(std::move(Name)), Sig(std::move(Sig)) {}

  py::object call(py::args Args) const {
    if (Args.size() != Sig.ArgTypes.size())
      throw py::type_error("function argument count does not match signature");

    py::tuple CoercedArgs(Args.size());
    py::module_ ctypes = py::module_::import("ctypes");
    for (std::size_t I = 0; I < Sig.ArgTypes.size(); ++I) {
      py::object Arg = py::reinterpret_borrow<py::object>(Args[I]);
      if (Sig.ArgTypes[I].Kind == PyType::Ptr)
        // Force pointer-typed arguments through `c_void_p` so ctypes does not
        // reinterpret Python integers as narrower scalar values.
        CoercedArgs[I] =
            ctypes.attr("c_void_p")(py::int_(ptrFromHostObject(Arg)));
      else
        CoercedArgs[I] = Arg;
    }
    return Callable(*CoercedArgs);
  }

  std::string repr() const {
    return "<proteus.Function name='" + Name +
           "' signature=" + signatureRepr(Sig) + ">";
  }
};

class Module {
  std::shared_ptr<ModuleBase> Impl;

public:
  explicit Module(std::shared_ptr<ModuleBase> Impl) : Impl(std::move(Impl)) {}

  Kernel getKernel(const std::string &Name, Signature Sig) {
    validateSignature(Sig);
    if (Sig.RetType.Kind != PyType::Void)
      throw py::type_error("kernel signatures must return proteus.void");
    return Kernel(Impl, Impl->getKernelAddress(Name), std::move(Sig), Name);
  }

  Function getFunction(const std::string &Name, Signature Sig) {
    validateSignature(Sig);
    if (!proteus::isHostTargetModel(Impl->getTargetModel()))
      throw py::value_error(
          "Target is a GPU model, cannot directly run functions, use launch()");

    py::module_ ctypes = py::module_::import("ctypes");
    // CFUNCTYPE expects the return type first, then the positional argument
    // types.
    py::tuple CTypeArgs(Sig.ArgTypes.size() + 1);
    CTypeArgs[0] = typeToCType(ctypes, Sig.RetType);
    for (std::size_t I = 0; I < Sig.ArgTypes.size(); ++I)
      CTypeArgs[I + 1] = typeToCType(ctypes, Sig.ArgTypes[I]);

    py::object FuncType = ctypes.attr("CFUNCTYPE")(*CTypeArgs);
    // Wrap the JIT symbol address in a Python callable with the requested
    // signature.
    py::object Callable = FuncType(
        py::int_(reinterpret_cast<uintptr_t>(Impl->getFunctionAddress(Name))));
    return Function(Impl, std::move(Callable), Name, std::move(Sig));
  }

  uintptr_t getFunctionAddress(const std::string &Name) {
    return reinterpret_cast<uintptr_t>(Impl->getFunctionAddress(Name));
  }
};

Module compile(py::object Source, const std::string &Frontend,
               const std::string &Target,
               const std::vector<std::string> &ExtraArgs,
               const std::string &Compiler, bool Verify) {
  // Frontend selection is validated here so the binding presents one
  // entrypoint.
  std::string Code = readSource(std::move(Source));
  std::shared_ptr<ModuleBase> Impl;
  if (Frontend == "cpp") {
    Impl = createCppModule(Target, Code, ExtraArgs, Compiler);
  } else if (Frontend == "llvmir") {
    if (Compiler != "clang")
      throw py::value_error("LLVMIR frontend does not support compiler='nvcc'");
    if (!ExtraArgs.empty())
      throw py::value_error("LLVMIR frontend does not support extra_args");
    Impl = createLLVMIRModule(Target, Code);
  } else if (Frontend == "mlir") {
    if (Compiler != "clang")
      throw py::value_error("MLIR frontend does not support compiler='nvcc'");
    if (!ExtraArgs.empty())
      throw py::value_error("MLIR frontend does not support extra_args");
#if PROTEUS_ENABLE_MLIR
    Impl = createMLIRModule(Target, Code);
#else
    throw py::value_error("MLIR frontend requires PROTEUS_ENABLE_MLIR");
#endif
  } else {
    throw py::value_error("frontend must be 'cpp', 'llvmir', or 'mlir'");
  }
  Impl->compile(Verify);
  return Module(std::move(Impl));
}

} // namespace
} // namespace proteus_python

using namespace proteus_python;

PYBIND11_MODULE(_proteus, M) {
  M.doc() = "Thin Python bindings for Proteus JIT frontends";

  // Expose the builtin scalar/pointer descriptors as module-level singletons.
  py::class_<Type>(M, "Type")
      .def("__call__", &makeSignature)
      .def("__repr__", &typeRepr)
      .def(
          "__eq__",
          [](const Type &LHS, const Type &RHS) { return LHS.Kind == RHS.Kind; },
          py::is_operator())
      .def(
          "__ne__",
          [](const Type &LHS, const Type &RHS) { return LHS.Kind != RHS.Kind; },
          py::is_operator())
      .def("__hash__",
           [](const Type &T) { return static_cast<py::ssize_t>(T.Kind); });

  py::class_<Signature>(M, "Signature")
      .def_property_readonly("restype",
                             [](const Signature &Sig) { return Sig.RetType; })
      .def_property_readonly("argtypes",
                             [](const Signature &Sig) {
                               py::tuple Result(Sig.ArgTypes.size());
                               for (std::size_t I = 0; I < Sig.ArgTypes.size();
                                    ++I)
                                 Result[I] = py::cast(Sig.ArgTypes[I]);
                               return Result;
                             })
      .def("__repr__", &signatureRepr);

  M.attr("void") = Type{PyType::Void};
  M.attr("i8") = Type{PyType::I8};
  M.attr("i32") = Type{PyType::I32};
  M.attr("i64") = Type{PyType::I64};
  M.attr("u32") = Type{PyType::U32};
  M.attr("u64") = Type{PyType::U64};
  M.attr("f32") = Type{PyType::F32};
  M.attr("f64") = Type{PyType::F64};
  M.attr("ptr") = Type{PyType::Ptr};

  // Modules own compiled code; kernels capture both the code and arg schema.
  py::class_<Module>(M, "Module")
      .def("get_kernel", &Module::getKernel, py::arg("name"), py::kw_only(),
           py::arg("signature"))
      .def("get_function", &Module::getFunction, py::arg("name"), py::kw_only(),
           py::arg("signature"))
      .def("get_function_address", &Module::getFunctionAddress,
           py::arg("name"));

  py::class_<Function>(M, "Function")
      .def("__call__", &Function::call)
      .def("__repr__", &Function::repr);

  py::class_<Kernel>(M, "Kernel")
      .def("__repr__", &Kernel::repr)
      .def("launch", &Kernel::launch, py::arg("grid"), py::arg("block"),
           py::arg("args"), py::arg("shmem") = 0,
           py::arg("stream") = py::none());

  M.def("compile", &compile, py::arg("source"), py::arg("frontend"),
        py::arg("target") = "host",
        py::arg("extra_args") = std::vector<std::string>{},
        py::arg("compiler") = "clang", py::arg("verify") = false);

  bindLLVMModule(M);

  M.attr("has_cuda") =
#if PROTEUS_ENABLE_CUDA
      true;
#else
      false;
#endif
  M.attr("has_hip") =
#if PROTEUS_ENABLE_HIP
      true;
#else
      false;
#endif
  M.attr("has_mlir") =
#if PROTEUS_ENABLE_MLIR
      true;
#else
      false;
#endif
}
