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

kernel void LOS_prepare1
(
    // матрица
    global const real *mat,
    global const int *aptr,
    global const int *jptr,
    const int n,
    // правая часть, массив будет использован для хранения z
    global real *f,
    // передаётся нач. приближ., записывается результат
    global const real *x,
    // вспомогательные массивы
    global real *r
)
{
    uint idx = get_global_id(0);

    r[idx] = MSRMulSingle(mat, aptr, jptr, n, x); // r = Ax
    r[idx] = f[idx] - r[idx]; // r = f - Ax

    f[idx] = r[idx]; // z = r.clone()
}

kernel void LOS_prepare2
(
    // матрица
    global const real *mat,
    global const int *aptr,
    global const int *jptr,
    const int n,
    // вспомогательные массивы
    global const real *r,
    global real *p
)
{
    uint idx = get_global_id(0);

    p[idx] = MSRMulSingle(mat, aptr, jptr, n, r); // p = Ar
}

kernel void LOS_xr
(
    global const float *f,
    global const float *p,
    const real alpha,
    global float *x,
    global float *r
)
{
    uint i = get_global_id(0);

    x[i] += (alpha * f[i]);
    r[i] -= (alpha * p[i]);
}

kernel void LOS_fp
(
    global const float *r,
    global const float *ar,
    const real beta,
    global float *f,
    global float *p
)
{
    uint i = get_global_id(0);

    f[i] = r[i] + beta * f[i];
    p[i] = ar[i] + beta * p[i];
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
    global real *r_hat,
    global real *p,
    global const real *f,
    global const real *x
)
{
    uint i = get_global_id(0);
    
    r[i] = f[i] - MSRMulSingle(mat, aptr, jptr, n, x);
    r_hat[i] = r[i];
    p[i] = r[i];
}

kernel void BiCGSTAB_hs
(
    global real *h,
    global real *s,
    global const real *p,
    global const real *nu,
    global const real *x,
    global const real *r,
    const real alpha
)
{
    uint i = get_global_id(0);
    
    h[i] = x[i] + alpha * p[i];
    s[i] = r[i] - alpha * nu[i];
}

kernel void BiCGSTAB_xr
(
    global real *x,
    global real *r,
    global const real *h,
    global const real *s,
    global const real *t,
    const real w
)
{
    uint i = get_global_id(0);
    
    x[i] = h[i] + w * s[i];
    r[i] = s[i] - w * t[i];
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