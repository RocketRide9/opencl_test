#define real float
#define real4 float4

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
    size_t row = get_global_id(0);

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
    size_t idx = get_global_id(0);

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
    size_t idx = get_global_id(0);

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
    size_t i = get_global_id(0);

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
    size_t i = get_global_id(0);

    // f[i] = mad(beta, f[i], r[i]);
    // p[i] = mad(beta, p[i], ar[i]);
    f[i] = r[i] + beta * f[i];
    p[i] = ar[i] + beta * p[i];
}

/*
kernel void LOS
(
    // матрица
    global const real *mat,
    global const int *aptr,
    global const int *jptr,
    const int n,
    // правая часть, массив будет использован для хранения z
    global real *f,
    // передаётся нач. приближ., записывается результат
    global real *res,
    // вспомогательные массивы
    global real *r,
    global real *p,
    global real *ar
)
{
    const float EPS = 1e-5;
    const int MAX_ITER = 1e+5;

    size_t idx = get_global_id(0);

    r[idx] = MSRMulSingle(mat, aptr, jptr, n, res); // r = Ax
    r[idx] = f[idx] - r[idx]; // r = f - Ax

    global real *z = f; // переименовать f в z
    z[idx] = r[idx]; // z = r.clone()

    barrier(CLK_LOCAL_MEM_FENCE);
    p[idx] = MSRMulSingle(mat, aptr, jptr, n, r); // p = Ar

    int iter = 0;
    // r должен быть посчитан полностью, перед вычислением произведения
    barrier(CLK_LOCAL_MEM_FENCE);
    real rr = dot(r, r, n);
    for (;iter < MAX_ITER && fabs(rr) > EPS; iter++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        real pp = dot(p, p, n);
        barrier(CLK_LOCAL_MEM_FENCE);
        real alpha = dot(p, r, n) / pp;

        res[idx] += alpha*z[idx];
        r[idx]   -= alpha*p[idx];

        barrier(CLK_LOCAL_MEM_FENCE);
        ar[idx] = MSRMulSingle(mat, aptr, jptr, n, r); // ar = Ar
        barrier(CLK_LOCAL_MEM_FENCE);
        real par = dot(p, ar, n);
        real beta = -(par) / pp;

        z[idx] = r[idx] + beta*z[idx]; // calc z
        p[idx] = ar[idx] + beta*p[idx]; // calc p

        rr -= alpha*alpha*pp;
    }
    if (idx == 0) {
        if (fabs(rr) > EPS)
        {
            printf("Точность не достигнута\n");
        }
        printf("Кол-во итераций: %d\n", iter);
    }
}
*/
