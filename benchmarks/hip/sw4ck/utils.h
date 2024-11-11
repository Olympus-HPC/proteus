#ifndef __UTILS_H
#define __UTILS_H

#include <hip/hip_runtime.h>

// Use double precision for accuracy 
#define float_sw4 double

void curvilinear4sg_ci(
    int ifirst, int ilast, 
    int jfirst, int jlast, 
    int kfirst, int klast,
    float_sw4* d_u, 
    float_sw4* d_mu,
    float_sw4* d_lambda,
    float_sw4* d_met,
    float_sw4* d_jac,
    float_sw4* d_lu, 
    int* onesided,
    float_sw4* d_acof, 
    float_sw4* d_bope,
    float_sw4* d_ghcof, 
    float_sw4* d_acof_no_gp,
    float_sw4* d_ghcof_no_gp, 
    float_sw4* d_strx,
    float_sw4* d_stry, 
    int nk, char op);

#endif
