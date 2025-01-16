// RUN: rm -rf .proteus
// RUN: ./scale100-gvar.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./scale100-gvar.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus

#include "../gpu_common.h"
#include <cstdio>
#include <cstdlib>

extern __device__ int gvar0;

__global__ void foo0(int *, int *, int);
__global__ void foo1(int *, int *, int);
__global__ void foo2(int *, int *, int);
__global__ void foo3(int *, int *, int);
__global__ void foo4(int *, int *, int);
__global__ void foo5(int *, int *, int);
__global__ void foo6(int *, int *, int);
__global__ void foo7(int *, int *, int);
__global__ void foo8(int *, int *, int);
__global__ void foo9(int *, int *, int);
__global__ void foo10(int *, int *, int);
__global__ void foo11(int *, int *, int);
__global__ void foo12(int *, int *, int);
__global__ void foo13(int *, int *, int);
__global__ void foo14(int *, int *, int);
__global__ void foo15(int *, int *, int);
__global__ void foo16(int *, int *, int);
__global__ void foo17(int *, int *, int);
__global__ void foo18(int *, int *, int);
__global__ void foo19(int *, int *, int);
__global__ void foo20(int *, int *, int);
__global__ void foo21(int *, int *, int);
__global__ void foo22(int *, int *, int);
__global__ void foo23(int *, int *, int);
__global__ void foo24(int *, int *, int);
__global__ void foo25(int *, int *, int);
__global__ void foo26(int *, int *, int);
__global__ void foo27(int *, int *, int);
__global__ void foo28(int *, int *, int);
__global__ void foo29(int *, int *, int);
__global__ void foo30(int *, int *, int);
__global__ void foo31(int *, int *, int);
__global__ void foo32(int *, int *, int);
__global__ void foo33(int *, int *, int);
__global__ void foo34(int *, int *, int);
__global__ void foo35(int *, int *, int);
__global__ void foo36(int *, int *, int);
__global__ void foo37(int *, int *, int);
__global__ void foo38(int *, int *, int);
__global__ void foo39(int *, int *, int);
__global__ void foo40(int *, int *, int);
__global__ void foo41(int *, int *, int);
__global__ void foo42(int *, int *, int);
__global__ void foo43(int *, int *, int);
__global__ void foo44(int *, int *, int);
__global__ void foo45(int *, int *, int);
__global__ void foo46(int *, int *, int);
__global__ void foo47(int *, int *, int);
__global__ void foo48(int *, int *, int);
__global__ void foo49(int *, int *, int);
__global__ void foo50(int *, int *, int);
__global__ void foo51(int *, int *, int);
__global__ void foo52(int *, int *, int);
__global__ void foo53(int *, int *, int);
__global__ void foo54(int *, int *, int);
__global__ void foo55(int *, int *, int);
__global__ void foo56(int *, int *, int);
__global__ void foo57(int *, int *, int);
__global__ void foo58(int *, int *, int);
__global__ void foo59(int *, int *, int);
__global__ void foo60(int *, int *, int);
__global__ void foo61(int *, int *, int);
__global__ void foo62(int *, int *, int);
__global__ void foo63(int *, int *, int);
__global__ void foo64(int *, int *, int);
__global__ void foo65(int *, int *, int);
__global__ void foo66(int *, int *, int);
__global__ void foo67(int *, int *, int);
__global__ void foo68(int *, int *, int);
__global__ void foo69(int *, int *, int);
__global__ void foo70(int *, int *, int);
__global__ void foo71(int *, int *, int);
__global__ void foo72(int *, int *, int);
__global__ void foo73(int *, int *, int);
__global__ void foo74(int *, int *, int);
__global__ void foo75(int *, int *, int);
__global__ void foo76(int *, int *, int);
__global__ void foo77(int *, int *, int);
__global__ void foo78(int *, int *, int);
__global__ void foo79(int *, int *, int);
__global__ void foo80(int *, int *, int);
__global__ void foo81(int *, int *, int);
__global__ void foo82(int *, int *, int);
__global__ void foo83(int *, int *, int);
__global__ void foo84(int *, int *, int);
__global__ void foo85(int *, int *, int);
__global__ void foo86(int *, int *, int);
__global__ void foo87(int *, int *, int);
__global__ void foo88(int *, int *, int);
__global__ void foo89(int *, int *, int);
__global__ void foo90(int *, int *, int);
__global__ void foo91(int *, int *, int);
__global__ void foo92(int *, int *, int);
__global__ void foo93(int *, int *, int);
__global__ void foo94(int *, int *, int);
__global__ void foo95(int *, int *, int);
__global__ void foo96(int *, int *, int);
__global__ void foo97(int *, int *, int);
__global__ void foo98(int *, int *, int);
__global__ void foo99(int *, int *, int);

int main() {
    int *a = nullptr;
    gpuErrCheck(gpuMallocManaged(&a, sizeof(int)*100));
    int *b = nullptr;
    gpuErrCheck(gpuMallocManaged(&b, sizeof(int)*100));
    for(int i=0; i<100; ++i) {
        a[i] = 0;
        b[i] = 1;
    }

    int num_blocks = std::max(1, 100/256);
    
    foo0<<<num_blocks, 256>>>(a, b, 100);
    foo1<<<num_blocks, 256>>>(a, b, 100);
    foo2<<<num_blocks, 256>>>(a, b, 100);
    foo3<<<num_blocks, 256>>>(a, b, 100);
    foo4<<<num_blocks, 256>>>(a, b, 100);
    foo5<<<num_blocks, 256>>>(a, b, 100);
    foo6<<<num_blocks, 256>>>(a, b, 100);
    foo7<<<num_blocks, 256>>>(a, b, 100);
    foo8<<<num_blocks, 256>>>(a, b, 100);
    foo9<<<num_blocks, 256>>>(a, b, 100);
    foo10<<<num_blocks, 256>>>(a, b, 100);
    foo11<<<num_blocks, 256>>>(a, b, 100);
    foo12<<<num_blocks, 256>>>(a, b, 100);
    foo13<<<num_blocks, 256>>>(a, b, 100);
    foo14<<<num_blocks, 256>>>(a, b, 100);
    foo15<<<num_blocks, 256>>>(a, b, 100);
    foo16<<<num_blocks, 256>>>(a, b, 100);
    foo17<<<num_blocks, 256>>>(a, b, 100);
    foo18<<<num_blocks, 256>>>(a, b, 100);
    foo19<<<num_blocks, 256>>>(a, b, 100);
    foo20<<<num_blocks, 256>>>(a, b, 100);
    foo21<<<num_blocks, 256>>>(a, b, 100);
    foo22<<<num_blocks, 256>>>(a, b, 100);
    foo23<<<num_blocks, 256>>>(a, b, 100);
    foo24<<<num_blocks, 256>>>(a, b, 100);
    foo25<<<num_blocks, 256>>>(a, b, 100);
    foo26<<<num_blocks, 256>>>(a, b, 100);
    foo27<<<num_blocks, 256>>>(a, b, 100);
    foo28<<<num_blocks, 256>>>(a, b, 100);
    foo29<<<num_blocks, 256>>>(a, b, 100);
    foo30<<<num_blocks, 256>>>(a, b, 100);
    foo31<<<num_blocks, 256>>>(a, b, 100);
    foo32<<<num_blocks, 256>>>(a, b, 100);
    foo33<<<num_blocks, 256>>>(a, b, 100);
    foo34<<<num_blocks, 256>>>(a, b, 100);
    foo35<<<num_blocks, 256>>>(a, b, 100);
    foo36<<<num_blocks, 256>>>(a, b, 100);
    foo37<<<num_blocks, 256>>>(a, b, 100);
    foo38<<<num_blocks, 256>>>(a, b, 100);
    foo39<<<num_blocks, 256>>>(a, b, 100);
    foo40<<<num_blocks, 256>>>(a, b, 100);
    foo41<<<num_blocks, 256>>>(a, b, 100);
    foo42<<<num_blocks, 256>>>(a, b, 100);
    foo43<<<num_blocks, 256>>>(a, b, 100);
    foo44<<<num_blocks, 256>>>(a, b, 100);
    foo45<<<num_blocks, 256>>>(a, b, 100);
    foo46<<<num_blocks, 256>>>(a, b, 100);
    foo47<<<num_blocks, 256>>>(a, b, 100);
    foo48<<<num_blocks, 256>>>(a, b, 100);
    foo49<<<num_blocks, 256>>>(a, b, 100);
    foo50<<<num_blocks, 256>>>(a, b, 100);
    foo51<<<num_blocks, 256>>>(a, b, 100);
    foo52<<<num_blocks, 256>>>(a, b, 100);
    foo53<<<num_blocks, 256>>>(a, b, 100);
    foo54<<<num_blocks, 256>>>(a, b, 100);
    foo55<<<num_blocks, 256>>>(a, b, 100);
    foo56<<<num_blocks, 256>>>(a, b, 100);
    foo57<<<num_blocks, 256>>>(a, b, 100);
    foo58<<<num_blocks, 256>>>(a, b, 100);
    foo59<<<num_blocks, 256>>>(a, b, 100);
    foo60<<<num_blocks, 256>>>(a, b, 100);
    foo61<<<num_blocks, 256>>>(a, b, 100);
    foo62<<<num_blocks, 256>>>(a, b, 100);
    foo63<<<num_blocks, 256>>>(a, b, 100);
    foo64<<<num_blocks, 256>>>(a, b, 100);
    foo65<<<num_blocks, 256>>>(a, b, 100);
    foo66<<<num_blocks, 256>>>(a, b, 100);
    foo67<<<num_blocks, 256>>>(a, b, 100);
    foo68<<<num_blocks, 256>>>(a, b, 100);
    foo69<<<num_blocks, 256>>>(a, b, 100);
    foo70<<<num_blocks, 256>>>(a, b, 100);
    foo71<<<num_blocks, 256>>>(a, b, 100);
    foo72<<<num_blocks, 256>>>(a, b, 100);
    foo73<<<num_blocks, 256>>>(a, b, 100);
    foo74<<<num_blocks, 256>>>(a, b, 100);
    foo75<<<num_blocks, 256>>>(a, b, 100);
    foo76<<<num_blocks, 256>>>(a, b, 100);
    foo77<<<num_blocks, 256>>>(a, b, 100);
    foo78<<<num_blocks, 256>>>(a, b, 100);
    foo79<<<num_blocks, 256>>>(a, b, 100);
    foo80<<<num_blocks, 256>>>(a, b, 100);
    foo81<<<num_blocks, 256>>>(a, b, 100);
    foo82<<<num_blocks, 256>>>(a, b, 100);
    foo83<<<num_blocks, 256>>>(a, b, 100);
    foo84<<<num_blocks, 256>>>(a, b, 100);
    foo85<<<num_blocks, 256>>>(a, b, 100);
    foo86<<<num_blocks, 256>>>(a, b, 100);
    foo87<<<num_blocks, 256>>>(a, b, 100);
    foo88<<<num_blocks, 256>>>(a, b, 100);
    foo89<<<num_blocks, 256>>>(a, b, 100);
    foo90<<<num_blocks, 256>>>(a, b, 100);
    foo91<<<num_blocks, 256>>>(a, b, 100);
    foo92<<<num_blocks, 256>>>(a, b, 100);
    foo93<<<num_blocks, 256>>>(a, b, 100);
    foo94<<<num_blocks, 256>>>(a, b, 100);
    foo95<<<num_blocks, 256>>>(a, b, 100);
    foo96<<<num_blocks, 256>>>(a, b, 100);
    foo97<<<num_blocks, 256>>>(a, b, 100);
    foo98<<<num_blocks, 256>>>(a, b, 100);
    foo99<<<num_blocks, 256>>>(a, b, 100);
    gpuErrCheck(gpuDeviceSynchronize());

    bool vecaddSuccess = true;
    for(int i=0; i<100; ++i)
        if(a[i] != 200) {
            vecaddSuccess = false;
            break;
        }

    bool gvarSuccess = true;
    int host_gvar;
    int n_gvar = 0;
    
    do {
        
        gpuErrCheck(gpuMemcpyFromSymbol(&host_gvar, gvar0, sizeof(int), /* offset */0, gpuMemcpyDeviceToHost));
        if(host_gvar != 101) {
            gvarSuccess = false;
            break;
        }
        n_gvar++;
        
    } while(0);
    

    if(vecaddSuccess && gvarSuccess)
        fprintf(stdout, "Verification successful\n");
    else {
        if(!vecaddSuccess)
            fprintf(stdout, "Vecadd failed\n");
        if(!gvarSuccess)
            fprintf(stdout, "Gvar failed gvar%d = %d != 102\n", n_gvar, host_gvar);
        fprintf(stdout, "Verification failed\n");
    }
    return 0;
}

// CHECK: Verification successful

// CHECK-COUNT-100: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 100
// CHECK-SECOND: JitStorageCache hits 100 total 100

