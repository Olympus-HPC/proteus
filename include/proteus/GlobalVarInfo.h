#pragma once
#include <cstdint>

namespace proteus {
struct GlobalVarInfo {
  const void *HostAddr;
  const void *DevAddr;
  uint64_t VarSize;
  GlobalVarInfo(const void *HostAddr, const void *DevAddr, uint64_t VarSize)
      : HostAddr(HostAddr), DevAddr(DevAddr), VarSize(VarSize) {}
};
} // namespace proteus
