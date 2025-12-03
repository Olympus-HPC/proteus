#ifndef PROTEUS_FRONTEND_LOOPUNROLLER_HPP
#define PROTEUS_FRONTEND_LOOPUNROLLER_HPP

#include <optional>

#include <llvm/IR/BasicBlock.h>

namespace proteus {

// Handles loop unroll configuration and metadata attachment.
class LoopUnroller {
  bool Enabled = false;
  std::optional<int> Count;

public:
  void enable();
  void enable(int Count);

  bool isEnabled() const;

  // Attach llvm.loop.unroll metadata to the back-edge branch in the latch.
  void attachMetadata(llvm::BasicBlock *LatchBB) const;
};

} // namespace proteus

#endif
