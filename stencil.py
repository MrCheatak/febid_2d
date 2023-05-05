import numpy as np
import pyopencl as cl
from pycl_test import cl_boilerplate
from timeit import default_timer as dt

def stencil1(a, c):
    print('Stencil operator on GPU ver.1 – generic')
    N = a.shape[0]
    index = np.arange(0, a.size).astype(np.int32)
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    index_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=index.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, index_dev, index)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    local_size = None
    global_size = a.shape
    start = dt()
    event = prog.matrix_mult1(queue, global_size, None, np.int32(N), a_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N * 5 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


def stencil2(a, c):
    print('Stencil operator on GPU ver.1 – generic')
    N = a.shape[0]
    index = np.arange(0, a.size).astype(np.int32)
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    index_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=index.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, index_dev, index)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    local_size = None
    global_size = a.shape
    start = dt()
    event = prog.matrix_mult1(queue, global_size, None, np.int32(N), a_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N * 5 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


if __name__ == '__main__':
    N = 2000
    a = np.random.randint(1, 10000, (N,N)).astype(np.float32)
    c = np.zeros_like(a)
    stencil1(a,c)
