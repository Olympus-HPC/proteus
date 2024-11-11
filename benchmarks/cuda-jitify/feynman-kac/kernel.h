__global__
#ifdef ENABLE_JIT
    __attribute__((annotate("jit", 5, 6, 7, 8)))
#endif
    void
    fk(const int ni, const int nj, int seed, const int N, const double a,
       const double b, const double h, const double rth,
       int *__restrict__ n_inside, double *__restrict__ err) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= ni && j <= nj) {
    double x =
        ((double)(nj - j) * (-a) + (double)(j - 1) * a) / (double)(nj - 1);

    double y =
        ((double)(ni - i) * (-b) + (double)(i - 1) * b) / (double)(ni - 1);

    double dx;
    double dy;
    double us;
    double ut;
    double vh;
    double vs;
    double x1;
    double x2;
    double w;
    double w_exact;
    double we;
    double wt;
    double chk = pow(x / a, 2.0) + pow(y / b, 2.0);

    if (1.0 < chk) {
      w_exact = 1.0;
      wt = 1.0;
    } else {
      atomicAdd(n_inside, 1);
      w_exact = exp(pow(x / a, 2.0) + pow(y / b, 2.0) - 1.0);
      wt = 0.0;
      for (int k = 0; k < N; k++) {
        x1 = x;
        x2 = y;
        w = 1.0;
        chk = 0.0;
        while (chk < 1.0) {
          ut = r8_uniform_01(&seed);
          if (ut < 1.0 / 2.0) {
            us = r8_uniform_01(&seed) - 0.5;
            if (us < 0.0)
              dx = -rth;
            else
              dx = rth;
          } else {
            dx = 0.0;
          }

          ut = r8_uniform_01(&seed);
          if (ut < 1.0 / 2.0) {
            us = r8_uniform_01(&seed) - 0.5;
            if (us < 0.0)
              dy = -rth;
            else
              dy = rth;
          } else {
            dy = 0.0;
          }
          vs = potential(a, b, x1, x2);
          x1 = x1 + dx;
          x2 = x2 + dy;

          vh = potential(a, b, x1, x2);

          we = (1.0 - h * vs) * w;
          w = w - 0.5 * h * (vh * we + vs * w);

          chk = pow(x1 / a, 2.0) + pow(x2 / b, 2.0);
        }
        wt += w;
      }
      wt /= (double)(N);
      atomicAdd(err, pow(w_exact - wt, 2.0));
    }
  }
}
