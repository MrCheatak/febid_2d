import pyopencl as cl
import numpy as np


def cl_boilerplate():
    # filename1 = 'kernel.cl'
    # filename2 = 'kernel_stencil.cl'
    # with open(filename1, 'r', encoding='utf-8') as f:
    #     kernel = ''.join(f.readlines())
    # with open('kernel_stencil.cl', 'r', encoding='utf-8') as f:
    #     kernel_stencil = ''.join(f.readlines())
    # kernel_jit = ''

    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()

    context = cl.Context(devices)

    # context = cl.create_some_context()
    # print(f'Using  {devices[0].name}')
    queue = cl.CommandQueue(context)

    # program1 = cl.Program(context, kernel).build()
    # program2 = cl.Program(context, kernel_stencil).build
    program = None

    return context, program, queue


def reaction_diffusion_jit(s, F, n0, tau, sigma, D, step, dt, global_size, local_size):
    text = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__constant double s = %(s).12f;
__constant double F = %(F).12f;
__constant double n0 = %(n0).12f;
__constant double tau = %(tau).12f;
__constant double sigma = %(sigma).12f;
__constant double D = %(D).12f;
__constant double step_x = %(step_x).12f;
__constant double dt = %(dt).12f;

__constant int global_size = %(global_size)d;
__constant int local_size = %(local_size)d;

__kernel void reaction_equation(__global float* array, __global float* array1, __global float* array2, int size) 
{
    float n;

    int gid = get_global_id(0);

    if (gid < size) 
    {
        n = array[gid];
        array[gid] += dt * (s * F * (1 - n / n0) - n / tau - n * sigma * array1[gid] + D * array2[gid] / step_x / step_x) * 1e-6;
    }
}

__kernel void stencil_operator(__global float* input, __global float* output, int size) 
{
    int gid = get_global_id(0);

    if (gid > 0 && gid < size - 1) 
    {
        output[gid] = input[gid - 1] - 2.0f * input[gid] + input[gid + 1];
    }
}

__kernel void stencil_operator_local_mem(__global float* input, __global float* output, int size) 
{
    // Shared memory for caching array elements
    __local float local_array[%(local_size)d];

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    int local_index = lid;

    // Load array elements to shared memory
    local_array[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Apply stencil operator to the shared memory elements
    __private float stencil_value = 0.0f;
    if (local_index > 0 && local_index < group_size - 1) {
        stencil_value = local_array[local_index - 1] - 2.0f * local_array[local_index] + local_array[local_index + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write the stencil output to global memory
    output[gid] = stencil_value;
        }


__kernel void stencil_rde(__global double* array, __global double* array1, int loops, int size, __global double* array2)
{
    // Shared memory for caching array elements
    __local double local_array[%(local_size)d+2];
    // __local double laplace[%(local_size)d];
    __local double f_local[%(local_size)d];
    __local int i;
    __local double Dsdt;
    __local double sFdt;
    __local double sFn0dt;
    __local double taudt;
    __private double stencil_value;
    __private double coeff;
    __local double n;
    __global double* readFrom = array;
    __global double* writeTo = array2;
    __global double* temp;
    
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    int local_index = lid + 1;
    
    // Load array elements to shared memory with margin
    //if (lid == 0) 
    //{
    //    local_array[0] = array[gid > 0 ? gid - 1 : 0];
    //}
    //local_array[local_index] = array[gid];
    //if (lid == group_size - 1) 
    //{
    //    local_array[local_index + 1] = array[gid < size - 1 ? gid + 1 : size - 1];
    //}
    f_local[lid] = array1[gid] * sigma * 1e-4;
    Dsdt = dt / step_x / step_x * D * 1e-6;
    sFdt = s * F * dt * 1e-6;
    sFn0dt = sFdt / n0;
    taudt = dt/ tau / 1e-4 * 1e-6;
    coeff = sFn0dt + taudt + f_local[lid] * dt * 1e-6;
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    for (i = 0; i < loops; i++)
    {   
        
        // Update shared memory for the next iteration
        //if (lid == 0) 
        //{   
        //    local_array[0] = array[gid > 0 ? gid - 1 : 0];
        //}
        //local_array[local_index] = array[gid];
        //if (lid == group_size) 
        //{
        //    local_array[local_index + 1] = array[gid < size - 1 ? gid + 1 : size - 1];
        //}
        
        //barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Stencil operation
        //if (gid >=0) 
        //{
            //stencil_value = local_array[local_index - 1] - 2 * local_array[local_index] + local_array[local_index + 1];
        //}
        if (gid > 0 && gid < size - 1)
        {
            stencil_value = readFrom[gid - 1] - 2 * readFrom[gid] + readFrom[gid + 1];
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // RDE calculation
        // n = local_array[local_index];
        n = readFrom[gid];
        // array[gid] += sFdt - n * sFn0dt - n * taudt - n * f_local[lid] * dt * 1e-6  + Dsdt * stencil_value;
        writeTo[gid] = n + sFdt - n * coeff + Dsdt * stencil_value;
        // array[gid] += dt * (s * F * (1 - n / n0) - n / tau - n * sigma * array1[gid] + D * array2[gid] / step_x / step_x) * 1e-6;
        // array[gid] += Dsdt * stencil_value;
               
        barrier(CLK_GLOBAL_MEM_FENCE);
        temp = readFrom;
        readFrom = writeTo;
        writeTo = temp;
    }
}


    """ % {'s': s, 'F': F, 'n0': n0, 'tau': tau, 'sigma': sigma, 'D': D, 'step_x': step, 'dt': dt,
           'global_size': global_size, 'local_size': local_size}
    return text


def reaction_diffusion_dimensionless_jit(tau_r, p_o, f0, step, dt, global_size, local_size):
    text = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__constant double tau_r = %(tau_r).12f;
__constant double p_o = %(p_o).12f;
__constant double f0 = %(f0).12f;
__constant double step_x = %(step_x).12f;
__constant double dt = %(dt).12f;

__constant int global_size = %(global_size)d;
__constant int local_size = %(local_size)d;

__kernel void stencil_rde(__global double* array, __global double* array1, int loops, int size, __global double* array2)
{
    // Shared memory for caching array elements
    __local double local_array[%(local_size)d+2];
    //__local double k_local[%(local_size)d];
    __local int i;
    __local double po2step2dt;
    __private double stencil_value;
    __private double k;
    __private double coeff;
    __local double n;
    __global double* readFrom = array;
    __global double* writeTo = array2;
    __global double* temp;

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    int local_index = lid + 1;

    // Load array elements to shared memory with margin
    k = (tau_r - 1) / f0 * array1[gid] + 1;
    po2step2dt = p_o * p_o / step_x / step_x * dt * 1e-6;
    coeff = k * dt * 1e-6;

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (i = 0; i < loops; i++)
    {   
        if (gid > 0 && gid < size - 1)
        {
            stencil_value = readFrom[gid - 1] - 2 * readFrom[gid] + readFrom[gid + 1];
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        // RDE calculation;
        n = readFrom[gid];
        writeTo[gid] = n + dt * 1e-6 - n * coeff + po2step2dt * stencil_value;

        barrier(CLK_GLOBAL_MEM_FENCE);
        temp = readFrom;
        readFrom = writeTo;
        writeTo = temp;
    }
}


    """ % {'tau_r':tau_r, 'p_o':p_o, 'f0':f0, 'step_x': step, 'dt': dt,
           'global_size': global_size, 'local_size': local_size}
    return text


def test_stencil():
    context, prog, queue = cl_boilerplate()
    data = np.arange(2, 1e3, 100)
    # data = np.log(data)
    result = np.zeros_like(data)
    result[0] = data[0]
    result[-1] = data[-1]
    data[...] = result[...]
    index = np.arange(0, data.size).astype(int)
    index = index[1:-1]
    data_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
    result_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=result.nbytes)
    index_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=index.nbytes)

    cl.enqueue_copy(queue, data_dev, data)
    cl.enqueue_copy(queue, index_dev, index)
    cl.enqueue_copy(queue, result_dev, result)

    for i in range(100000):
        prog.stencil(queue, index.shape, None, data_dev, index_dev, result_dev)
        temp = result_dev
        result_dev = data_dev
        data_dev = temp
        cl.enqueue_copy(queue, result, data_dev)

    cl.enqueue_copy(queue, result, result_dev)

    a = 0


if __name__ == '__main__':
    context, prog, queue = cl_boilerplate()
    n = np.zeros(100)
    x = np.full_like(n, 1)
    a = np.array([1.0])
    b = np.array([2.0])

    n_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=n.nbytes)
    x_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=x.nbytes)

    cl.enqueue_copy(queue, x_dev, x)

    for i in range(100):
        prog.simple_math(queue, n.shape, None, n_dev, a, b, x_dev)
        cl.enqueue_copy(queue, n, n_dev)
