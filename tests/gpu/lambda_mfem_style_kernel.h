#include "proteus/JitInterface.h"
#include "raja_mfem_style_launch.h"


__attribute__((noinline)) PROTEUS_HOST_DEVICE void PrintD1D(int D1D) { printf("D1D = %d\n", D1D); }
__attribute__((noinline)) PROTEUS_HOST_DEVICE void PrintQ1D(int Q1D) { printf("Q1D = %d\n", Q1D); }

template<int T_D1D = 0, int T_Q1D = 0>
inline void IntegralKernel(double FloatConst, int PD1D, int PQ1D)
{
   const int D1D = T_D1D ? T_D1D : PD1D;
   const int Q1D = T_Q1D ? T_Q1D : PQ1D;

   MockMfemInterface::forall(1, 1,
                             proteus::register_lambda([=,
                               D1D = proteus::jit_variable(D1D),
                               Q1D = proteus::jit_variable(Q1D),
                               FloatConst = proteus::jit_variable(FloatConst)]
                               PROTEUS_HOST_DEVICE (int, int)
   {
      for (int Q = 0; Q < Q1D; ++Q)
      {
        for (int D = 0; D < D1D; ++D)
        {
          if (FloatConst > 0.0)
            PrintD1D(D1D);
          else
            PrintQ1D(Q1D);
        }
      }
   }));
}

