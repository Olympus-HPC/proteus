#ifndef CURVILINEAR4SG_H
#define CURVILINEAR4SG_H

#include "utils.h"

template <int N>
class Range {
 public:
  Range(int istart, int iend) : start(istart), end(iend), tpb(N) {
    blocks = (end - start) / N;
    blocks = ((end - start) % N == 0) ? blocks : blocks + 1;
    invalid = false;
    if (blocks <= 0) invalid = true;
  };
  int start;
  int end;
  int blocks;
  int tpb;
  bool invalid;
};


#define ni      (ilast - ifirst + 1)
#define nij     (ni * (jlast - jfirst + 1))
#define nijk    (nij * (klast - kfirst + 1))
#define base    (-(ifirst + ni * jfirst + nij * kfirst))
#define base3   (base - nijk)
#define base4   (base - nijk)
#define ifirst0 (ifirst)
#define jfirst0 (jfirst)

#define mu(i, j, k) a_mu[base + (i) + ni * (j) + nij * (k)]
#define la(i, j, k) a_lambda[base + (i) + ni * (j) + nij * (k)]
#define jac(i, j, k) a_jac[base + (i) + ni * (j) + nij * (k)]
#define u(c, i, j, k) a_u[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define lu(c, i, j, k) a_lu[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define met(c, i, j, k) a_met[base4 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define strx(i) a_strx[i - ifirst0]
#define stry(j) a_stry[j - jfirst0]
#define acof(i, j, k) a_acof[(i - 1) + 6 * (j - 1) + 48 * (k - 1)]
#define bope(i, j) a_bope[i - 1 + 6 * (j - 1)]
#define ghcof(i) a_ghcof[i - 1]
#define acof_no_gp(i, j, k) a_acof_no_gp[(i - 1) + 6 * (j - 1) + 48 * (k - 1)]
#define ghcof_no_gp(i) a_ghcof_no_gp[i - 1]

#define i6 ((float_sw4)(1.0 / 6))
#define tf ((float_sw4)(0.75))
#define c1 ((float_sw4)(2.0 / 3))
#define c2 ((float_sw4)(-1.0 / 12))

#endif
