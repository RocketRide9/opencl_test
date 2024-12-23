#define real double
#define real4 double4

kernel void MSRMul(
    global const real *mat,
    global const int *aptr,
    global const int *jptr,
    const int n,
    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    int start = aptr[row];
    int stop = aptr[row + 1];
    real dot = mat[row]*v[row];
    for (int a = start; a < stop; a++)
    {
        dot += mat[a]*v[jptr[a-n]];
    }
    res[row] = dot;
}

/*
float MSCMulSingle
(
    global const real *mat,
    global const int *aptr,
    global const int *iptr,
    const int n,
    global const real *v
)
{
    size_t row = get_global_id(0);

    int start = aptr[row];
    int stop = aptr[row + 1];
    real dot = mat[row]*v[row];
    for (int a = start; a < stop; a++)
    {
        dot += mat[a]*v[iptr[a-n]];
    }
    return dot;
}
*/

float MSRMulSingle
(
    global const real *mat,
    global const int *aptr,
    global const int *jptr,
    const int n,
    global const real *v
)
{
    uint row = get_global_id(0);

    int start = aptr[row];
    int stop = aptr[row + 1];
    real dot = mat[row]*v[row];
    for (int a = start; a < stop; a++)
    {
        dot += mat[a]*v[jptr[a-n]];
    }
    return dot;
}

kernel void dot_kernel
(
    global const real4 *v1,
    global const real4 *v2,
    const int n,
    global real *res,
    local real *work
)
{
    // Compute partial dot product
    real sum = 0;
    for (int k=get_global_id(0);k<n;k+=get_global_size(0))
    {
        // sum += a[get_global_id(ROW_DIM)+m*k] * x[k];
        sum += dot(v1[k], v2[k]);
    }

    // Each thread stores its partial sum in WORK
    int cols = get_local_size(0); // initial cols in group
    // int ii = get_local_id(ROW_DIM); // local row index in group, 0<=ii<rows
    int jj = get_local_id(0); // block index in column, 0<=jj<cols
    work[jj] = sum;
    barrier(CLK_LOCAL_MEM_FENCE); // sync group

    // Reduce sums in log2(cols) steps
    while ( cols > 1 )
    {
        cols /= 2;
        if (jj < cols) work[jj] += work[jj+cols];
        barrier(CLK_LOCAL_MEM_FENCE); // sync group
    }

    // Write final result in Y
    if ( jj == 0 ) *res = work[0];
}

kernel void BiCGSTAB_prepare1
(
    // матрица
    global const real *mat,
    global const int *aptr,
    global const int *jptr,
    const int n,
    // вспомогательные массивы
    global real *r,
    global real *p,
    global const real *f,
    global const real *x
)
{
    uint i = get_global_id(0);
    
    r[i] = f[i] - MSRMulSingle(mat, aptr, jptr, n, x);
}

// y += a*x
kernel void BLAS_axpy
(
    const real a,
    global const real *x,
    global real *y
)
{
    uint i = get_global_id(0);
    
    y[i] += a * x[i];
}

// y *= a
kernel void BLAS_scale
(
    const real a,
    global real *y
)
{
    uint i = get_global_id(0);
    
    y[i] *= a;
}

kernel void BiCGSTAB_p
(
    global real *p,
    global const real *r,
    global const real *nu,
    const real w,
    const real beta
)
{
    uint i = get_global_id(0);
    
    p[i] = r[i] + beta * (p[i] - w*nu[i]);
}

// код взят отсюда https://github.com/CNugteren/CLBlast/blob/bd96941ac0633e8e7d09fd2475e0279be370b1e1/src/kernels/level1/xdot.opencl

#define WGS1 32
#define WGS2 32

kernel
void Xdot(const int n,
          const global real* restrict xgm,
          const global real* restrict ygm,
          global real* output)
{
    local real lm[WGS1];
    const int lid = get_local_id(0);
    const int wgid = get_group_id(0);
    const int num_groups = get_num_groups(0);
    // Performs multiplication and the first steps of the reduction
    real acc = 0;
    int id = wgid*WGS1 + lid;
    while (id < n) {
        real x = xgm[id];
        real y = ygm[id];
        acc += x * y;
        id += WGS1*num_groups;
    }
    lm[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Performs reduction in local memory
    for (int s=WGS1/2; s>0; s = s >> 1) {
        if (lid < s) {
            lm[lid] = lm[lid] + lm[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Stores the per-workgroup result
    if (lid == 0) {
        output[wgid] = lm[0];
    }
}


// The epilogue reduction kernel, performing the final bit of the sum operation. This kernel has to
// be launched with a single workgroup only.
kernel
void XdotEpilogue(const global real* restrict input,
                  global real* dot) {
    local real lm[WGS2];
    const int lid = get_local_id(0);
    
    // Performs the first step of the reduction while loading the data
    lm[lid] = input[lid] + input[lid + WGS2];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Performs reduction in local memory
    for (int s=WGS2/2; s>0; s=s>>1) {
        if (lid < s) {
            lm[lid] += lm[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Stores the final result
    if (lid == 0) {
        dot[0] = lm[0];
    }
}
