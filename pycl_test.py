import pyopencl as cl
import numpy as np


def cl_boilerplate():
    with open('kernel.cl', 'r', encoding='utf-8') as f:
        kernel = ''.join(f.readlines())

    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()

    context = cl.Context(devices)
    # context = cl.create_some_context()
    print(f'Using  {devices[0].name}')
    queue = cl.CommandQueue(context)

    program = cl.Program(context, kernel).build()

    return context, program, queue


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

