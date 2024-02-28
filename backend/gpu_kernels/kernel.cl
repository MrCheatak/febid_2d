__kernel void reaction_equation(__global double *n,
               const double s,
               const double F,
               const double n0,
               const double tau,
               const double sigma,
               __global const double *f,
               const double D,
               __global const double *n_D,
               const double step,
               const double dt)
{
    int i = get_global_id(0);
    double n_val = n[i];
    n[i] += (s*F*(1-n_val/n0) - n_val/tau - sigma*f[i]*n_val + n_D[i]*D/step/step)*dt;
}

__kernel void simple_math(__global double *n,
                          const double a, const double b,
                          __global double *x)
{
    int i = get_global_id(0);
    n[i] = n[i] + a - b * x[i];
}

__kernel void matrix_mult1(const int N,
                          __global double *a,
                          __global double *b,
                          __global double *c)
{
    int k;
    int i = get_global_id(0);
    int j = get_global_id(1);
    double tmp = 0.0f;
    for (k = 0; k < N; k++)
    {
        tmp += a[i*N+k] * b[k*N+j];
    }
    c[i*N+j] = tmp;
}

__kernel void matrix_mult1_float(const int N,
                          __global float *a,
                          __global float *b,
                          __global float *c)
{
    int k;
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0.0f;
    for (k = 0; k < N; k++)
    {
        tmp += a[i*N+k] * b[k*N+j];
    }
    c[i*N+j] = tmp;
}

__kernel void matrix_mult1_float_consts(const int N,
                          __global const float *a,
                          __global const float *b,
                          __global float *c)
{
    int k;
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0.0f;
    for (k = 0; k < N; k++)
    {
        tmp += a[i*N+k] * b[k*N+j];
    }
    c[i*N+j] = tmp;
}


__kernel void matrix_mult2_float(const int N,
                          __global float *a,
                          __global float *b,
                          __global float *c)
{
    int j, k;
    float tmp;
    int i = get_global_id(0);
    for (j = 0; j < N; j++)
    {
        tmp = 0.0f;
        for (k = 0; k < N; k++)
        {
            tmp += a[i*N+k] * b[k*N+j];
        }
        c[i*N+j] = tmp;
    }
}

__kernel void matrix_mult3_float(const int N,
                          __global float *a,
                          __global float *b,
                          __global float *c)
{
    int j, k;
    float tmp;
    float awrk[5000];
    int i = get_global_id(0);
    for (k = 0; k < N; k++)
    {
        awrk[k] = a[i*N+k];
    }
    for (j = 0; j < N; j++)
    {
        tmp = 0.0f;
        for (k = 0; k < N; k++)
        {
            tmp += awrk[k] * b[k*N+j];
        }
        c[i*N+j] = tmp;
    }
}


__kernel void matrix_mult4_float_const_tiles(int const N,
                                             __global const float *A,
                                             __global const float *B,
                                             __global float *C)
{
    short TSIZE = 16;
    short GRID_SIZE = 16;
    __local float Asub[16][16];
    __local float Bsub[16][16];

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);

    int a_row = TSIZE * by + ty;
    int a_col = TSIZE * tx;
    int b_row = TSIZE * by;
    int b_col = TSIZE * tx + tx;

    float sum = 0.0;
    for (int i = 0; i < GRID_SIZE; i += TSIZE) {
        if (a_row < N && a_col + i < N) {
            Asub[ty][tx + i] = A[a_row * N + a_col + i];
        } else {
            Asub[ty][tx + i] = 0.0;
        }

        if (b_row + i < N && b_col < N) {
            Bsub[ty + i][tx] = B[(b_row + i) * N + b_col];
        } else {
            Bsub[ty + i][tx] = 0.0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < TSIZE; p++) {
            sum += Asub[ty][p] * Bsub[p][tx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int c_row = TSIZE * by + ty;
    int c_col = TSIZE * bx + tx;
    if (c_row < N && c_col < N) {
        C[c_row * N + c_col] = sum;
    }
}


 __kernel void matrix_mult6_float_const_vector(int const N,
                                             __global const float *A,
                                             __global const float *B,
                                             __global float4 *C)
    {
        int tx = get_global_id(0);
        int ty = get_global_id(1);

        float4 sum = (float4)(0.0, 0.0, 0.0, 0.0);
        for (int i = 0; i < N; i++) {
            float4 a = A[ty * N + i];
            float4 b = B[i * N + tx];
            sum += a * b;
        }

        int idx = ty * N + tx;
        if (ty < N && tx < N) {
            C[idx] = sum;
        }
    }
