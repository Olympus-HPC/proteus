#include "raja_style_launch.h"

#include <utility>

namespace MockMfemInterface {
template <typename DBODY>
void RajaWrap(const int N, DBODY &&d_body)
{
  MockRajaInterface::forall(N, std::forward<DBODY>(d_body));
}

template <typename d_lambda>
inline void ForallWrap(const int N, d_lambda &&d_body)
{
  return RajaWrap(N, std::forward<d_lambda>(d_body));
}

template<typename lambda>
__attribute__((noinline)) void forall(int N, lambda &&body)
{
  ForallWrap(N, std::forward<lambda>(body));
}

template<typename lambda>
__attribute__((noinline)) void forall(int Nx, int Ny, lambda &&body)
{
  MockMfemInterface::forall(Nx * Ny, [=] PROTEUS_HOST_DEVICE (int idx)
  {
      int j = idx / Nx;
      int i = idx % Nx;
      body(i, j);
  });
}
}
