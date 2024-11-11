__global__ 
#ifdef ENABLE_JIT
__attribute__((annotate("jit", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)))
#endif
void kernel3(
    const int start0, const int N0, 
    const int start1, const int N1,
    const int start2, const int N2,
    const int ifirst, const int ilast,
    const int jfirst, const int jlast,
    const int kfirst, const int klast,
    const float_sw4 a1, const float_sw4 sgn,
    const float_sw4* __restrict__ a_u, 
    const float_sw4* __restrict__ a_mu,
    const float_sw4* __restrict__ a_lambda,
    const float_sw4* __restrict__ a_met,
    const float_sw4* __restrict__ a_jac,
          float_sw4* __restrict__ a_lu, 
    const float_sw4* __restrict__ a_acof, 
    const float_sw4* __restrict__ a_bope,
    const float_sw4* __restrict__ a_ghcof, 
    const float_sw4* __restrict__ a_acof_no_gp,
    const float_sw4* __restrict__ a_ghcof_no_gp, 
    const float_sw4* __restrict__ a_strx,
    const float_sw4* __restrict__ a_stry ) 
{
  int i = start0 + threadIdx.x + blockIdx.x * blockDim.x;
  int j = start1 + threadIdx.y + blockIdx.y * blockDim.y;
  int k = start2 + threadIdx.z + blockIdx.z * blockDim.z;
  if ((i < N0) && (j < N1) && (k < N2)) {
    float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
    float_sw4 istry = 1 / (stry(j));
    float_sw4 istrx = 1 / (strx(i));
    float_sw4 istrxy = istry * istrx;

    float_sw4 r2 = 0;
    // v-equation

    //      r1 = 0;
    // pp derivative (v)
    // 43 ops, tot=816
    float_sw4 cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) *
      met(1, i - 2, j, k) * strx(i - 2);
    float_sw4 cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) *
      met(1, i - 1, j, k) * strx(i - 1);
    float_sw4 cof3 =
      (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
    float_sw4 cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) *
      met(1, i + 1, j, k) * strx(i + 1);
    float_sw4 cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) *
      met(1, i + 2, j, k) * strx(i + 2);

    float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
    float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 *
      (mux1 * (u(2, i - 2, j, k) - u(2, i, j, k)) +
       mux2 * (u(2, i - 1, j, k) - u(2, i, j, k)) +
       mux3 * (u(2, i + 1, j, k) - u(2, i, j, k)) +
       mux4 * (u(2, i + 2, j, k) - u(2, i, j, k))) *
      istry;

    // qq derivative (v)
    // 53 ops, tot=869
    cof1 = (2 * mu(i, j - 2, k) + la(i, j - 2, k)) * met(1, i, j - 2, k) *
      met(1, i, j - 2, k) * stry(j - 2);
    cof2 = (2 * mu(i, j - 1, k) + la(i, j - 1, k)) * met(1, i, j - 1, k) *
      met(1, i, j - 1, k) * stry(j - 1);
    cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
      met(1, i, j, k) * stry(j);
    cof4 = (2 * mu(i, j + 1, k) + la(i, j + 1, k)) * met(1, i, j + 1, k) *
      met(1, i, j + 1, k) * stry(j + 1);
    cof5 = (2 * mu(i, j + 2, k) + la(i, j + 2, k)) * met(1, i, j + 2, k) *
      met(1, i, j + 2, k) * stry(j + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 *
      (mux1 * (u(2, i, j - 2, k) - u(2, i, j, k)) +
       mux2 * (u(2, i, j - 1, k) - u(2, i, j, k)) +
       mux3 * (u(2, i, j + 1, k) - u(2, i, j, k)) +
       mux4 * (u(2, i, j + 2, k) - u(2, i, j, k))) *
      istrx;

    // rr derivative (u)
    // 42 ops, tot=911
    cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
      met(3, i, j, k - 2);
    cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
      met(3, i, j, k - 1);
    cof3 =
      (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(3, i, j, k);
    cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
      met(3, i, j, k + 1);
    cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
      met(3, i, j, k + 2);

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 * (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +
        mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +
        mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +
        mux4 * (u(1, i, j, k + 2) - u(1, i, j, k)));

    // rr derivative (v)
    // 83 ops, tot=994
    cof1 = (2 * mu(i, j, k - 2) + la(i, j, k - 2)) * met(3, i, j, k - 2) *
      stry(j) * met(3, i, j, k - 2) * stry(j) +
      mu(i, j, k - 2) * (met(2, i, j, k - 2) * strx(i) *
          met(2, i, j, k - 2) * strx(i) +
          met(4, i, j, k - 2) * met(4, i, j, k - 2));
    cof2 = (2 * mu(i, j, k - 1) + la(i, j, k - 1)) * met(3, i, j, k - 1) *
      stry(j) * met(3, i, j, k - 1) * stry(j) +
      mu(i, j, k - 1) * (met(2, i, j, k - 1) * strx(i) *
          met(2, i, j, k - 1) * strx(i) +
          met(4, i, j, k - 1) * met(4, i, j, k - 1));
    cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(3, i, j, k) * stry(j) *
      met(3, i, j, k) * stry(j) +
      mu(i, j, k) *
      (met(2, i, j, k) * strx(i) * met(2, i, j, k) * strx(i) +
       met(4, i, j, k) * met(4, i, j, k));
    cof4 = (2 * mu(i, j, k + 1) + la(i, j, k + 1)) * met(3, i, j, k + 1) *
      stry(j) * met(3, i, j, k + 1) * stry(j) +
      mu(i, j, k + 1) * (met(2, i, j, k + 1) * strx(i) *
          met(2, i, j, k + 1) * strx(i) +
          met(4, i, j, k + 1) * met(4, i, j, k + 1));
    cof5 = (2 * mu(i, j, k + 2) + la(i, j, k + 2)) * met(3, i, j, k + 2) *
      stry(j) * met(3, i, j, k + 2) * stry(j) +
      mu(i, j, k + 2) * (met(2, i, j, k + 2) * strx(i) *
          met(2, i, j, k + 2) * strx(i) +
          met(4, i, j, k + 2) * met(4, i, j, k + 2));

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 *
      (mux1 * (u(2, i, j, k - 2) - u(2, i, j, k)) +
       mux2 * (u(2, i, j, k - 1) - u(2, i, j, k)) +
       mux3 * (u(2, i, j, k + 1) - u(2, i, j, k)) +
       mux4 * (u(2, i, j, k + 2) - u(2, i, j, k))) *
      istrxy;

    // rr derivative (w)
    // 43 ops, tot=1037
    cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(3, i, j, k - 2) *
      met(4, i, j, k - 2);
    cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(3, i, j, k - 1) *
      met(4, i, j, k - 1);
    cof3 =
      (mu(i, j, k) + la(i, j, k)) * met(3, i, j, k) * met(4, i, j, k);
    cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(3, i, j, k + 1) *
      met(4, i, j, k + 1);
    cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(3, i, j, k + 2) *
      met(4, i, j, k + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 *
      (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +
       mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +
       mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +
       mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *
      istrx;

    // pq-derivatives
    // 38 ops, tot=1075
    r2 +=
      c2 *
      (la(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(1, i + 2, j + 2, k) - u(1, i - 2, j + 2, k)) +
        c1 * (u(1, i + 1, j + 2, k) - u(1, i - 1, j + 2, k))) -
       la(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
       (c2 * (u(1, i + 2, j - 2, k) - u(1, i - 2, j - 2, k)) +
        c1 * (u(1, i + 1, j - 2, k) - u(1, i - 1, j - 2, k)))) +
      c1 *
      (la(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
       (c2 * (u(1, i + 2, j + 1, k) - u(1, i - 2, j + 1, k)) +
        c1 * (u(1, i + 1, j + 1, k) - u(1, i - 1, j + 1, k))) -
       la(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
       (c2 * (u(1, i + 2, j - 1, k) - u(1, i - 2, j - 1, k)) +
        c1 * (u(1, i + 1, j - 1, k) - u(1, i - 1, j - 1, k))));

    // qp-derivatives
    // 38 ops, tot=1113
    r2 +=
      c2 *
      (mu(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
       (c2 * (u(1, i + 2, j + 2, k) - u(1, i + 2, j - 2, k)) +
        c1 * (u(1, i + 2, j + 1, k) - u(1, i + 2, j - 1, k))) -
       mu(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
       (c2 * (u(1, i - 2, j + 2, k) - u(1, i - 2, j - 2, k)) +
        c1 * (u(1, i - 2, j + 1, k) - u(1, i - 2, j - 1, k)))) +
      c1 *
      (mu(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
       (c2 * (u(1, i + 1, j + 2, k) - u(1, i + 1, j - 2, k)) +
        c1 * (u(1, i + 1, j + 1, k) - u(1, i + 1, j - 1, k))) -
       mu(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
       (c2 * (u(1, i - 1, j + 2, k) - u(1, i - 1, j - 2, k)) +
        c1 * (u(1, i - 1, j + 1, k) - u(1, i - 1, j - 1, k))));

    // pr-derivatives
    // 82 ops, tot=1195
    r2 +=
      c2 *
      ((la(i, j, k + 2)) * met(3, i, j, k + 2) *
       met(1, i, j, k + 2) *
       (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +
        c1 * (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) +
       mu(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(2, i + 2, j, k + 2) - u(2, i - 2, j, k + 2)) +
        c1 * (u(2, i + 1, j, k + 2) - u(2, i - 1, j, k + 2))) *
       strx(i) * istry -
       ((la(i, j, k - 2)) * met(3, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +
         c1 * (u(1, i + 1, j, k - 2) - u(1, i - 1, j, k - 2))) +
        mu(i, j, k - 2) * met(2, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(2, i + 2, j, k - 2) - u(2, i - 2, j, k - 2)) +
         c1 * (u(2, i + 1, j, k - 2) - u(2, i - 1, j, k - 2))) *
        strx(i) * istry)) +
      c1 *
      ((la(i, j, k + 1)) * met(3, i, j, k + 1) *
       met(1, i, j, k + 1) *
       (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +
        c1 * (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) +
       mu(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
       (c2 * (u(2, i + 2, j, k + 1) - u(2, i - 2, j, k + 1)) +
        c1 * (u(2, i + 1, j, k + 1) - u(2, i - 1, j, k + 1))) *
       strx(i) * istry -
       (la(i, j, k - 1) * met(3, i, j, k - 1) *
        met(1, i, j, k - 1) *
        (c2 * (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +
         c1 * (u(1, i + 1, j, k - 1) - u(1, i - 1, j, k - 1))) +
        mu(i, j, k - 1) * met(2, i, j, k - 1) *
        met(1, i, j, k - 1) *
        (c2 * (u(2, i + 2, j, k - 1) - u(2, i - 2, j, k - 1)) +
         c1 * (u(2, i + 1, j, k - 1) - u(2, i - 1, j, k - 1))) *
        strx(i) * istry));

    // rp derivatives
    // 82 ops, tot=1277
    r2 +=
      c2 *
      ((mu(i + 2, j, k)) * met(3, i + 2, j, k) *
       met(1, i + 2, j, k) *
       (c2 * (u(1, i + 2, j, k + 2) - u(1, i + 2, j, k - 2)) +
        c1 * (u(1, i + 2, j, k + 1) - u(1, i + 2, j, k - 1))) +
       mu(i + 2, j, k) * met(2, i + 2, j, k) * met(1, i + 2, j, k) *
       (c2 * (u(2, i + 2, j, k + 2) - u(2, i + 2, j, k - 2)) +
        c1 * (u(2, i + 2, j, k + 1) - u(2, i + 2, j, k - 1))) *
       strx(i + 2) * istry -
       (mu(i - 2, j, k) * met(3, i - 2, j, k) *
        met(1, i - 2, j, k) *
        (c2 * (u(1, i - 2, j, k + 2) - u(1, i - 2, j, k - 2)) +
         c1 * (u(1, i - 2, j, k + 1) - u(1, i - 2, j, k - 1))) +
        mu(i - 2, j, k) * met(2, i - 2, j, k) *
        met(1, i - 2, j, k) *
        (c2 * (u(2, i - 2, j, k + 2) - u(2, i - 2, j, k - 2)) +
         c1 * (u(2, i - 2, j, k + 1) - u(2, i - 2, j, k - 1))) *
        strx(i - 2) * istry)) +
      c1 *
      ((mu(i + 1, j, k)) * met(3, i + 1, j, k) *
       met(1, i + 1, j, k) *
       (c2 * (u(1, i + 1, j, k + 2) - u(1, i + 1, j, k - 2)) +
        c1 * (u(1, i + 1, j, k + 1) - u(1, i + 1, j, k - 1))) +
       mu(i + 1, j, k) * met(2, i + 1, j, k) * met(1, i + 1, j, k) *
       (c2 * (u(2, i + 1, j, k + 2) - u(2, i + 1, j, k - 2)) +
        c1 * (u(2, i + 1, j, k + 1) - u(2, i + 1, j, k - 1))) *
       strx(i + 1) * istry -
       (mu(i - 1, j, k) * met(3, i - 1, j, k) *
        met(1, i - 1, j, k) *
        (c2 * (u(1, i - 1, j, k + 2) - u(1, i - 1, j, k - 2)) +
         c1 * (u(1, i - 1, j, k + 1) - u(1, i - 1, j, k - 1))) +
        mu(i - 1, j, k) * met(2, i - 1, j, k) *
        met(1, i - 1, j, k) *
        (c2 * (u(2, i - 1, j, k + 2) - u(2, i - 1, j, k - 2)) +
         c1 * (u(2, i - 1, j, k + 1) - u(2, i - 1, j, k - 1))) *
        strx(i - 1) * istry));

    // qr derivatives
    // 130 ops, tot=1407
    r2 +=
      c2 *
      (mu(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j - 2, k + 2)) +
        c1 * (u(1, i, j + 1, k + 2) - u(1, i, j - 1, k + 2))) +
       (2 * mu(i, j, k + 2) + la(i, j, k + 2)) *
       met(3, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +
        c1 * (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) *
       stry(j) * istrx +
       mu(i, j, k + 2) * met(4, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(3, i, j + 2, k + 2) - u(3, i, j - 2, k + 2)) +
        c1 * (u(3, i, j + 1, k + 2) - u(3, i, j - 1, k + 2))) *
       istrx -
       (mu(i, j, k - 2) * met(2, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(1, i, j + 2, k - 2) - u(1, i, j - 2, k - 2)) +
         c1 * (u(1, i, j + 1, k - 2) - u(1, i, j - 1, k - 2))) +
        (2 * mu(i, j, k - 2) + la(i, j, k - 2)) *
        met(3, i, j, k - 2) * met(1, i, j, k - 2) *
        (c2 * (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +
         c1 * (u(2, i, j + 1, k - 2) - u(2, i, j - 1, k - 2))) *
        stry(j) * istrx +
        mu(i, j, k - 2) * met(4, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(3, i, j + 2, k - 2) - u(3, i, j - 2, k - 2)) +
         c1 * (u(3, i, j + 1, k - 2) - u(3, i, j - 1, k - 2))) *
        istrx)) +
        c1 *
        (mu(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(1, i, j + 2, k + 1) - u(1, i, j - 2, k + 1)) +
          c1 * (u(1, i, j + 1, k + 1) - u(1, i, j - 1, k + 1))) +
         (2 * mu(i, j, k + 1) + la(i, j, k + 1)) *
         met(3, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +
          c1 * (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) *
         stry(j) * istrx +
         mu(i, j, k + 1) * met(4, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(3, i, j + 2, k + 1) - u(3, i, j - 2, k + 1)) +
          c1 * (u(3, i, j + 1, k + 1) - u(3, i, j - 1, k + 1))) *
         istrx -
         (mu(i, j, k - 1) * met(2, i, j, k - 1) *
          met(1, i, j, k - 1) *
          (c2 * (u(1, i, j + 2, k - 1) - u(1, i, j - 2, k - 1)) +
           c1 * (u(1, i, j + 1, k - 1) - u(1, i, j - 1, k - 1))) +
          (2 * mu(i, j, k - 1) + la(i, j, k - 1)) *
          met(3, i, j, k - 1) * met(1, i, j, k - 1) *
          (c2 * (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +
           c1 * (u(2, i, j + 1, k - 1) - u(2, i, j - 1, k - 1))) *
          stry(j) * istrx +
          mu(i, j, k - 1) * met(4, i, j, k - 1) *
          met(1, i, j, k - 1) *
          (c2 * (u(3, i, j + 2, k - 1) - u(3, i, j - 2, k - 1)) +
           c1 * (u(3, i, j + 1, k - 1) - u(3, i, j - 1, k - 1))) *
          istrx));

    // rq derivatives
    // 130 ops, tot=1537
    r2 +=
      c2 *
      (la(i, j + 2, k) * met(2, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j + 2, k - 2)) +
        c1 * (u(1, i, j + 2, k + 1) - u(1, i, j + 2, k - 1))) +
       (2 * mu(i, j + 2, k) + la(i, j + 2, k)) *
       met(3, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j + 2, k - 2)) +
        c1 * (u(2, i, j + 2, k + 1) - u(2, i, j + 2, k - 1))) *
       stry(j + 2) * istrx +
       la(i, j + 2, k) * met(4, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(3, i, j + 2, k + 2) - u(3, i, j + 2, k - 2)) +
        c1 * (u(3, i, j + 2, k + 1) - u(3, i, j + 2, k - 1))) *
       istrx -
       (la(i, j - 2, k) * met(2, i, j - 2, k) *
        met(1, i, j - 2, k) *
        (c2 * (u(1, i, j - 2, k + 2) - u(1, i, j - 2, k - 2)) +
         c1 * (u(1, i, j - 2, k + 1) - u(1, i, j - 2, k - 1))) +
        (2 * mu(i, j - 2, k) + la(i, j - 2, k)) *
        met(3, i, j - 2, k) * met(1, i, j - 2, k) *
        (c2 * (u(2, i, j - 2, k + 2) - u(2, i, j - 2, k - 2)) +
         c1 * (u(2, i, j - 2, k + 1) - u(2, i, j - 2, k - 1))) *
        stry(j - 2) * istrx +
        la(i, j - 2, k) * met(4, i, j - 2, k) *
        met(1, i, j - 2, k) *
        (c2 * (u(3, i, j - 2, k + 2) - u(3, i, j - 2, k - 2)) +
         c1 * (u(3, i, j - 2, k + 1) - u(3, i, j - 2, k - 1))) *
        istrx)) +
        c1 *
        (la(i, j + 1, k) * met(2, i, j + 1, k) * met(1, i, j + 1, k) *
         (c2 * (u(1, i, j + 1, k + 2) - u(1, i, j + 1, k - 2)) +
          c1 * (u(1, i, j + 1, k + 1) - u(1, i, j + 1, k - 1))) +
         (2 * mu(i, j + 1, k) + la(i, j + 1, k)) *
         met(3, i, j + 1, k) * met(1, i, j + 1, k) *
         (c2 * (u(2, i, j + 1, k + 2) - u(2, i, j + 1, k - 2)) +
          c1 * (u(2, i, j + 1, k + 1) - u(2, i, j + 1, k - 1))) *
         stry(j + 1) * istrx +
         la(i, j + 1, k) * met(4, i, j + 1, k) * met(1, i, j + 1, k) *
         (c2 * (u(3, i, j + 1, k + 2) - u(3, i, j + 1, k - 2)) +
          c1 * (u(3, i, j + 1, k + 1) - u(3, i, j + 1, k - 1))) *
         istrx -
         (la(i, j - 1, k) * met(2, i, j - 1, k) *
          met(1, i, j - 1, k) *
          (c2 * (u(1, i, j - 1, k + 2) - u(1, i, j - 1, k - 2)) +
           c1 * (u(1, i, j - 1, k + 1) - u(1, i, j - 1, k - 1))) +
          (2 * mu(i, j - 1, k) + la(i, j - 1, k)) *
          met(3, i, j - 1, k) * met(1, i, j - 1, k) *
          (c2 * (u(2, i, j - 1, k + 2) - u(2, i, j - 1, k - 2)) +
           c1 * (u(2, i, j - 1, k + 1) - u(2, i, j - 1, k - 1))) *
          stry(j - 1) * istrx +
          la(i, j - 1, k) * met(4, i, j - 1, k) *
          met(1, i, j - 1, k) *
          (c2 * (u(3, i, j - 1, k + 2) - u(3, i, j - 1, k - 2)) +
           c1 * (u(3, i, j - 1, k + 1) - u(3, i, j - 1, k - 1))) *
          istrx));

    // 4 ops, tot=1541
    lu(2, i, j, k) = a1 * lu(2, i, j, k) + sgn * r2 * ijac;
  }
}

