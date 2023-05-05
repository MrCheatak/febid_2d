import numpy as np
import pyopencl as cl
from pycl_test import cl_boilerplate
from timeit import default_timer as dt


def matrix_mult_cpu(a , b, c):
    print('Matrix multiplication on CPU')
    start = dt()
    c = a.dot(b)
    result = dt() - start
    print(f'Calculation took {result:.4f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')
    return c


def matrix_mult1(a , b, c):
    print('Matrix multiplication on GPU ver.1 – generic')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    start = dt()
    event = prog.matrix_mult1(queue, a.shape, None, np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


def matrix_mult1_float(a , b, c):
    print('Matrix multiplication on GPU ver.1 ‒ floats instead of doubles')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    start = dt()
    event = prog.matrix_mult1_float(queue, a.shape, None, np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


def matrix_mult1_float_const(a , b, c):
    print('Matrix multiplication on GPU ver.1 – using \'const\' for unchanged arrays')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    start = dt()
    event = prog.matrix_mult1_float_consts(queue, a.shape, None, np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


def matrix_mult1_float_const_16x16(a , b, c):
    print('Matrix multiplication on GPU ver.1 – using 16x16 work groups')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    start = dt()
    event = prog.matrix_mult1_float_consts(queue, a.shape, (25, 25), np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


def matrix_mult2_float(a , b, c):
    print('Matrix multiplication on GPU ver.2 ‒ managing work group size manually')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    start = dt()
    event = prog.matrix_mult2_float(queue, (N, 1), (100, 1), np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


def matrix_mult3_float(a , b, c):
    print('Matrix multiplication on GPU ver.3 – using local work group memory')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    start = dt()
    event = prog.matrix_mult3_float(queue, (N, 1), (100, 1), np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


def matrix_mult4_float_const_tiles(a , b, c):
    print('Matrix multiplication on GPU ver.4 – using tiling strategy (by chatGPT)')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    print(f'Memory transfer took {result:.4f} s')

    local_size = (16,16)
    global_size = ((N+15)//16*16, (N+15)//16*16)
    start = dt()
    event = prog.matrix_mult4_float_const_tiles(queue, global_size, local_size, np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    print(f'Memory transfer back took {result:.3f} s')
    return c


def matrix_mult5_float_const_shared_mem(a , b, c):
    print('Matrix multiplication on GPU ver.5 – using vectors (by chatGPT')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    local_size = (16, 16)
    global_size = (int(np.ceil(N/16))*16, int(np.ceil(N/16))*16)
    start = dt()
    event = prog.matrix_mult5_float_const_shared_mem(queue, global_size, local_size, np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c


def matrix_mult6_float_const_vector(a , b, c):
    print('Matrix multiplication on GPU ver.5 – using vectors (by chatGPT')
    context, prog, queue = cl_boilerplate()

    a_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=c.nbytes)

    start = dt()
    cl.enqueue_copy(queue, a_dev, a)
    cl.enqueue_copy(queue, b_dev, b)
    result = dt() - start
    # print(f'Memory transfer took {result:.4f} s')

    local_size = (16, 16)
    global_size = (int(np.ceil(N/16))*16, int(np.ceil(N/16))*16)
    start = dt()
    event = prog.matrix_mult6_float_const_vector(queue, global_size, local_size, np.int32(N), a_dev, b_dev, c_dev)
    event.wait()
    result = dt() - start
    print(f'Calculation took {result:.3f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')

    start = dt()
    cl.enqueue_copy(queue, c, c_dev)
    result = dt() - start
    # print(f'Memory transfer back took {result:.3f} s')
    return c




if __name__ == '__main__':
    N = 20000
    a = np.random.randint(1, 10, (N, N)).astype(np.float32)
    b = np.random.randint(1, 10, (N, N)).astype(np.float32)
    c = np.zeros_like(a, dtype=np.float32)
    c = matrix_mult_cpu(a, b, c)
    # c = matrix_mult1(a, b, c)
    # c = matrix_mult1_float(a, b, c)
    # c = matrix_mult1_float_const(a, b, c)
    # c = matrix_mult1_float_const_16x16(a, b, c)
    # c = matrix_mult2_float(a, b, c)
    # c = matrix_mult3_float(a, b, c)
    c = matrix_mult4_float_const_tiles(a, b, c)
    # c = matrix_mult5_float_const_vector(a, b, c)
