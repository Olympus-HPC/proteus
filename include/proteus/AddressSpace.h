#ifndef PROTEUS_ADDRESS_SPACE_H
#define PROTEUS_ADDRESS_SPACE_H

namespace proteus {

enum class AddressSpace : unsigned int {
  DEFAULT = 0,
  GLOBAL = 1,
  SHARED = 3,
  CONSTANT = 4,
  LOCAL = 5,
};

inline constexpr unsigned toLLVM(AddressSpace AS) {
  return static_cast<unsigned>(AS);
}

} // namespace proteus

#endif // PROTEUS_ADDRESS_SPACE_H
