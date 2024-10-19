#define real float
#define ROW_DIM 0
#define COL_DIM 1
#pragma OPENCL EXTENSION cl_ext_float_atomics : enable

kernel void mul(global const real *mat,
                global const real *v,
                global real *r,
                int n,                          // columns
                int m,                          // rows
                local real *group_work)
{
    size_t gi = get_global_id(ROW_DIM);
    size_t gj = get_global_id(COL_DIM);

    size_t ii = get_local_id(ROW_DIM);
    size_t jj = get_local_id(COL_DIM);
    size_t cols = get_local_size(COL_DIM);
    size_t rows = get_local_size(ROW_DIM);

    real sum = 0;
    for (int k = gj; k < n; k += cols)
    {
        sum += mat[gi + k*m] * v[k];
    }
    group_work[ii + rows*jj] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    while (cols > 1)
    {
        cols /= 2;
        if (jj < cols)
        {
            group_work[ii + jj*rows] += group_work[ii + (jj+cols)*rows];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (jj == 0)
    {
        r[gi] = group_work[ii];
    }
}

kernel void mul_simple
                (global const real *mat,
                global const real *v,
                global real *r,
                int n,                      // columns
                int m)                      // rows
{
    int i = get_global_id (0);

    real sum = 0;
    for (int j = 0; j < n; j++)
    {
        sum += mat[i + m*j] * v[j];
    }
    r[i] = sum;
}

kernel void gemv3(global const real *mat,
                  global const real *x,
		          global real *r,
		          int n, int m,
		          local real *work)
{
    // Load a slice of X in WORK, using all available threads
    int ncols = n / get_global_size(COL_DIM); // nb values to load
    int col0 = ncols * get_global_id(COL_DIM); // first value to load
    for (int k=0; k<ncols; k+=get_local_size(ROW_DIM))
    {
        int col = k+get_local_id(ROW_DIM);
        if (col < ncols)
        {
            work[col] = x[col0+col];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // sync group

    // Compute partial dot product
    real sum = (real)0;
    for (int k=0; k<ncols; k++)
    {
        sum += mat[get_global_id(ROW_DIM)+m*(col0+k)] * work[k];
    }

    // Store in Y (P columns per row)
    r[get_global_id(ROW_DIM)+m*get_global_id(COL_DIM)] = sum;
}

// Reduce M = get_global_size(0) rows of P values in matrix Y.
// Stores the result in first column of Y.
kernel void reduce_rows(global real *y, int m, int p)
{
    int row = get_global_id(0);
    real sum = (real)0;
    for (int col=0; col < p; col++)
    {
        sum += y[row + m*col];
    }
    y[row] = sum;
}
