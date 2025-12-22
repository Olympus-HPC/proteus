#ifndef PROTEUS_FRONTEND_TYPETRAITS_HPP
#define PROTEUS_FRONTEND_TYPETRAITS_HPP

#include <type_traits>

namespace proteus {

// Type alias to remove cv-qualifiers and references.
template <typename T>
using clean_t = std::remove_cv_t<std::remove_reference_t<T>>;

// NOLINTBEGIN(readability-identifier-naming)

// True if T (after removing references) is an arithmetic type.
template <typename T>
inline constexpr bool is_arithmetic_unref_v =
    std::is_arithmetic_v<std::remove_reference_t<T>>;

// True if T (after removing references) is a pointer type.
template <typename T>
inline constexpr bool is_pointer_unref_v =
    std::is_pointer_v<std::remove_reference_t<T>>;

// True if T is a scalar arithmetic type (not a pointer or array).
// Used for the primary Var specialization for numeric types.
template <typename T>
inline constexpr bool is_scalar_arithmetic_v =
    std::is_arithmetic_v<std::remove_reference_t<T>> &&
    !std::is_pointer_v<std::remove_reference_t<T>> &&
    !std::is_array_v<std::remove_reference_t<T>>;

// NOLINTEND(readability-identifier-naming)

} // namespace proteus

#endif // PROTEUS_FRONTEND_TYPETRAITS_HPP
