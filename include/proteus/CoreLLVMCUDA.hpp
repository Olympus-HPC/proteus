#ifndef PROTEUS_CORE_LLVM_CUDA_HPP
#define PROTEUS_CORE_LLVM_CUDA_HPP

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Module.h>

#include "proteus/UtilsCUDA.h"

namespace proteus {

using namespace llvm;

namespace detail {

static const SmallVector<StringRef> &gridDimXFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.nctaid.x"};
  return Names;
}

static const SmallVector<StringRef> &gridDimYFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.nctaid.y"};
  return Names;
}

static const SmallVector<StringRef> &gridDimZFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.nctaid.z"};
  return Names;
}

static const SmallVector<StringRef> &blockDimXFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.ntid.x"};
  return Names;
}

static const SmallVector<StringRef> &blockDimYFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.ntid.y"};
  return Names;
}

static const SmallVector<StringRef> &blockDimZFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.ntid.z"};
  return Names;
}

static const SmallVector<StringRef> &blockIdxXFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.ctaid.x"};
  return Names;
}

static const SmallVector<StringRef> &blockIdxYFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.ctaid.y"};
  return Names;
}

static const SmallVector<StringRef> &blockIdxZFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.ctaid.z"};
  return Names;
}

static const SmallVector<StringRef> &threadIdxXFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.tid.x"};
  return Names;
}

static const SmallVector<StringRef> &threadIdxYFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.tid.y"};
  return Names;
}

static const SmallVector<StringRef> &threadIdxZFnName() {
  static SmallVector<StringRef> Names = {"llvm.nvvm.read.ptx.sreg.tid.z"};
  return Names;
}

} // namespace detail

static inline void setLaunchBoundsForKernel(Module &M, Function &F,
                                            size_t GridSize, int BlockSize) {
  NamedMDNode *NvvmAnnotations = M.getNamedMetadata("nvvm.annotations");
  assert(NvvmAnnotations && "Expected non-null nvvm.annotations metadata");
  // TODO: fix hardcoded 1024 as the maximum, by reading device
  // properties.
  // TODO: set min GridSize.
  int MaxThreads = std::min(1024, BlockSize);
  auto *FuncMetadata = ConstantAsMetadata::get(&F);
  auto *MaxntidxMetadata = MDString::get(M.getContext(), "maxntidx");
  auto *MaxThreadsMetadata = ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(M.getContext()), MaxThreads));

  // Replace if the metadata exists.
  for (auto *MetadataNode : NvvmAnnotations->operands()) {
    // Expecting 3 operands ptr, desc, i32 value.
    assert(MetadataNode->getNumOperands() == 3);

    auto *PtrMetadata = MetadataNode->getOperand(0).get();
    auto *DescMetadata = MetadataNode->getOperand(1).get();
    if (PtrMetadata == FuncMetadata && MaxntidxMetadata == DescMetadata) {
      MetadataNode->replaceOperandWith(2, MaxThreadsMetadata);
      return;
    }
  }

  // Otherwise create the metadata and insert.
  Metadata *MDVals[] = {FuncMetadata, MaxntidxMetadata, MaxThreadsMetadata};
  NvvmAnnotations->addOperand(MDNode::get(M.getContext(), MDVals));
}
static inline void codegenPTX(Module &M, StringRef DeviceArch,
                              SmallVectorImpl<char> &PTXStr) {
  // TODO: It is possbile to use PTX directly through the CUDA PTX JIT
  // interface. Maybe useful if we can re-link globals using the CUDA API.
  // Check this reference for PTX JIT caching:
  // https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
  // Interesting env vars: CUDA_CACHE_DISABLE, CUDA_CACHE_MAXSIZE,
  // CUDA_CACHE_PATH, CUDA_FORCE_PTX_JIT.
  auto TMExpected = proteus::detail::createTargetMachine(M, DeviceArch);
  if (!TMExpected)
    FATAL_ERROR(toString(TMExpected.takeError()));

  std::unique_ptr<TargetMachine> TM = std::move(*TMExpected);
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(
      reinterpret_cast<LLVMTargetMachine *>(TM.get()));

  raw_svector_ostream PTXOS(PTXStr);
  TM->addPassesToEmitFile(PM, PTXOS, nullptr, CGFT_AssemblyFile,
                          /* DisableVerify */ false, MMIWP);

  PM.run(M);
}

static inline std::unique_ptr<MemoryBuffer>
codegenObject(Module &M, StringRef DeviceArch,
              SmallPtrSetImpl<void *> &GlobalLinkedBinaries) {
  SmallVector<char, 4096> PTXStr;
  size_t BinSize;

  codegenPTX(M, DeviceArch, PTXStr);
  PTXStr.push_back('\0');

  nvPTXCompilerHandle PTXCompiler;
  proteusNvPTXCompilerErrCheck(
      nvPTXCompilerCreate(&PTXCompiler, PTXStr.size(), PTXStr.data()));
  std::string ArchOpt = ("--gpu-name=" + DeviceArch).str();
  std::string RDCOption = "";
  if (!GlobalLinkedBinaries.empty())
    RDCOption = "-c";
#if PROTEUS_ENABLE_DEBUG
  const char *CompileOptions[] = {ArchOpt.c_str(), "--verbose",
                                  RDCOption.c_str()};
  size_t NumCompileOptions = 2 + (RDCOption.empty() ? 0 : 1);
#else
  const char *CompileOptions[] = {ArchOpt.c_str(), RDCOption.c_str()};
  size_t NumCompileOptions = 1 + (RDCOption.empty() ? 0 : 1);
#endif
  proteusNvPTXCompilerErrCheck(
      nvPTXCompilerCompile(PTXCompiler, NumCompileOptions, CompileOptions));
  proteusNvPTXCompilerErrCheck(
      nvPTXCompilerGetCompiledProgramSize(PTXCompiler, &BinSize));
  auto ObjBuf = WritableMemoryBuffer::getNewUninitMemBuffer(BinSize);
  proteusNvPTXCompilerErrCheck(
      nvPTXCompilerGetCompiledProgram(PTXCompiler, ObjBuf->getBufferStart()));
#if PROTEUS_ENABLE_DEBUG
  {
    size_t LogSize;
    proteusNvPTXCompilerErrCheck(
        nvPTXCompilerGetInfoLogSize(PTXCompiler, &LogSize));
    auto Log = std::make_unique<char[]>(LogSize);
    proteusNvPTXCompilerErrCheck(
        nvPTXCompilerGetInfoLog(PTXCompiler, Log.get()));
    Logger::logs("proteus") << "=== nvPTXCompiler Log\n" << Log.get() << "\n";
  }
#endif
  proteusNvPTXCompilerErrCheck(nvPTXCompilerDestroy(&PTXCompiler));

  std::unique_ptr<MemoryBuffer> FinalObjBuf;
  if (!GlobalLinkedBinaries.empty()) {
    CUlinkState CULinkState;
    proteusCuErrCheck(cuLinkCreate(0, nullptr, nullptr, &CULinkState));
    for (auto *Ptr : GlobalLinkedBinaries) {
      // We do not know the size of the binary but the CUDA API just needs a
      // non-zero argument.
      proteusCuErrCheck(cuLinkAddData(CULinkState, CU_JIT_INPUT_FATBINARY, Ptr,
                                      1, "", 0, 0, 0));
    }

    // Again using a non-zero argument, though we can get the size from the ptx
    // compiler.
    proteusCuErrCheck(cuLinkAddData(
        CULinkState, CU_JIT_INPUT_FATBINARY,
        static_cast<void *>(ObjBuf->getBufferStart()), 1, "", 0, 0, 0));

    void *BinOut;
    size_t BinSize;
    proteusCuErrCheck(cuLinkComplete(CULinkState, &BinOut, &BinSize));
    FinalObjBuf = std::move(MemoryBuffer::getMemBufferCopy(
        StringRef{static_cast<char *>(BinOut), BinSize}));
  } else {
    FinalObjBuf = std::move(ObjBuf);
  }

  return std::move(FinalObjBuf);
}

} // namespace proteus

#endif