// The lower bound to MKX is kpe - kps + 1 
#ifndef MKX
  #error(need a defined constant MKX that is static number of levels)
#endif

#define bi   blockIdx.x
#define bj   blockIdx.y
#define bx   blockDim.x
#define by   blockDim.y
#define ti   threadIdx.x
#define tj   threadIdx.y

# define ix   (ime-ims+1)
# define jx   (jme-jms+1)
# define kx   (kme-kms+1)


// basic indexing macros. indices are always given as global indices
// in undecompsed Domain(ids:ide,jds:jde)
//
// That is, given IJ (global index), the global Index mapped to
// a local index on a Patch(0:nx-1,0:ny-1) in Device Memory as:
//
// I - (ips-ims) + nx * ( J - (jps-jms) )
//
// where ips is the global index of the start of the patch (the -1 is
// for translating from WRF fortran indices).
//
// The global index I is mapped to a local index on a GPU Block's
// shared memory (0:bx-1, 0:by-1) as:
//
// I - (ips-ims) - bi * bx  +  by * ( J - (jps-jms) - bj * by )
//
// Where bi is the index into the GPU Block, and bx is the
// GPU Block Width.

// global to patch index converter
#define GtoP(i,p,P)      ((i)-(p)+(P))
#define GtoB(i,n,N,p,P)  ((i)-(p)+(P)-(n)*(N))

// thread index to local memory index = i + bi * bx + ips - ims
#define TtoP(i,a,b,c,d)  ((i)+(a)*(b)+(c)-(d))

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

// basic indexing macros
#define I2(i,j,m) ((i)+((j)*(m)))
#define I3(i,j,m,k,n) (I2(i,j,m)+((k)*(m)*(n)))

// index into a patch stored on device memory - 1
# define P2(i,j)    I2(TtoP(i,bi,bx,ips,ims),TtoP(j,bj,by,jps,jms),ime-ims+1)
# define P3(i,k,j)  I3(TtoP(i,bi,bx,ips,ims),k,ime-ims+1,TtoP(j,bj,by,jps,jms),kme-kms+1)

// index into a block stored on shared memory
# define S2(i,j)    I2(i,j,bx)
//# define S3(i,k,j)  I3(i,k,bx,j,kme-kms+1)
//# define S3(i,k,j)  I3(i,j,bx,k,by)
# define S3(i,k,j)  I3(k,i,kx,j,bx)

#define ig (TtoP(ti,bi,bx,ips,ims))
#define jg (TtoP(tj,bj,by,jps,jms))


