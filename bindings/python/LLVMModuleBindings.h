#ifndef PROTEUS_PYTHON_LLVM_MODULE_BINDINGS_H
#define PROTEUS_PYTHON_LLVM_MODULE_BINDINGS_H

namespace pybind11 {
class module_;
}

namespace proteus_python {

void bindLLVMModule(pybind11::module_ &M);

} // namespace proteus_python

#endif // PROTEUS_PYTHON_LLVM_MODULE_BINDINGS_H
