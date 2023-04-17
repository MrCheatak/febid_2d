__kernel void stencil(__global double *n,
                      __global const int *index,
                      __global double *n_out)
{
        int ind = get_global_id(0);
        int i = index[ind];
        n_out[i] = n[i+1] + n[i-1] - 2 * n[i];
}

__kernel void mult2(__global const double *a,
                  __global double *b)
{
    int gid = get_global_id(0);
    b[gid] = convert_int(gid);
}

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
    double tmp = 0.0f;
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
    float awrk[2500];
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
