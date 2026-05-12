// clang-format off
// RUN: %build/lambda_signature | %FILECHECK %s
// clang-format on

#include <iostream>
#include <type_traits>

#include <proteus/Frontend/TypeTraits.h>

int main() {
  auto l1 = [](int x, double y) -> long {
    return static_cast<long>(x + y);
  };
  using Sig1 = proteus::callable_signature_t<decltype(l1)>;
  static_assert(std::is_same_v<Sig1, long(int, double)>);
  static_assert(proteus::function_traits<Sig1>::arity == 2);

  auto l2 = [](int x) noexcept { return x + 1; };
  using Sig2 = proteus::callable_signature_t<decltype(l2)>;
  static_assert(std::is_same_v<Sig2, int(int) noexcept>);

  auto generic = [](auto x, auto y) { return x + y; };
  using Sig3 = proteus::call_signature_t<decltype(generic), int, float>;
  static_assert(std::is_same_v<Sig3, float(int, float)>);

  std::cout << "ok\n";
  return 0;
}

// clang-format off
// CHECK: ok
// clang-format on
