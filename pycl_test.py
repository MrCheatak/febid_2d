import pyopencl as cl
import numpy as np


def cl_boilerplate():
    with open('kernel.cl', 'r', encoding='utf-8') as f:
        kernel = ''.join(f.readlines())
    with open('kernel_stencil.cl', 'r', encoding='utf-8') as f:
        kernel_stencil = ''.join(f.readlines())
    kernel_jit = ''

    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()

    context = cl.Context(devices)
    # context = cl.create_some_context()
    print(f'Using  {devices[0].name}')
    queue = cl.CommandQueue(context)

    program1 = cl.Program(context, kernel).build()
    program2 = cl.Program(context, kernel_stencil).build()

    return context, (program1,program2), queue


def reaction_diffusion_jit(s, F, n0, tau, sigma, D, step, dt, local_size):
    text = """
__constant float s = %f;
__constant float F = %f;
__constant float n0 = %f;
__constant float tau = %f;
__constant float sigma = %f;
__constant float D = %f;
__constant float step_x = %f;
__constant float dt = %f;

__constant int local_size = %d;

__kernel void reaction_equation(__global float* array, __global float* array1, __global float* array2, int size) 
{
    __local float n;

    int gid = get_global_id(0);

    if (gid > 0 && gid < size - 1) 
    {
        n = array[gid];
        array[gid] = dt * (s * F * (1 - n / n0) - n / tau - n * sigma * array1[gid] + D * array2[gid] / step_x / step_x);
    }
}

__kernel void stencil_operator(__global float* input, __global float* output, int size) 
{
    int gid = get_global_id(0);

    if (gid > 0 && gid < size - 1) {
        output[gid] = input[gid - 1] - 2.0f * input[gid] + input[gid + 1];
    }
}

__kernel void stencil_operator_local_mem(__global float* input, __global float* output, int size) {
            // Shared memory for caching array elements
            __local float local_array[local_size];

            int gid = get_global_id(0);
            int lid = get_local_id(0);
            int group_size = get_local_size(0);
            int local_index = lid;

            // Load array elements to shared memory
            local_array[lid] = input[gid];
            barrier(CLK_LOCAL_MEM_FENCE);

            // Apply stencil operator to the shared memory elements
            float stencil_value = 0.0f;
            if (local_index > 0 && local_index < group_size - 1) {
                stencil_value = local_array[local_index - 1] - 2.0f * local_array[local_index] + local_array[local_index + 1];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Write the stencil output to global memory
            output[gid] = stencil_value;
        }



    """ % (s, F, n0, tau, sigma, D, step, dt, local_size)
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

