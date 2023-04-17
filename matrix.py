import numpy as np
import pyopencl as cl
from pycl_test import cl_boilerplate
from timeit import default_timer as dt


def matrix_mult_cpu(N=2000):
    print('Matrix multiplication on CPU')
    a = np.random.randint(1, 10, (N, N)).astype(np.float32)
    b = np.random.randint(1, 10, (N, N)).astype(np.float32)

    start = dt()
    c = a.dot(b)
    result = dt() - start
    print(f'Calculation took {result:.4f} s')
    print(f'{N ** 3 / result / 1e6} MFLOPS \n')
    return c

def matrix_mult1(N=2000):
    print('Matrix multiplication on GPU ver.1')
    context, prog, queue = cl_boilerplate()
    # a = np.arange(1, 26, dtype=float).reshape(5, 5)
    # b = np.arange(1, 51, 2, dtype=float).reshape(5, 5)
    a = np.random.randint(1, 10, (N, N)).astype(float)
    b = np.random.randint(1, 10, (N, N)).astype(float)
    c = np.zeros_like(a, dtype=float)

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

def matrix_mult1_float(N=2000):
    print('Matrix multiplication on GPU ver.1')
    context, prog, queue = cl_boilerplate()
    # a = np.arange(1, 26, dtype=float).reshape(5, 5)
    # b = np.arange(1, 51, 2, dtype=float).reshape(5, 5)
    a = np.random.randint(1, 10, (N, N)).astype(np.float32)
    b = np.random.randint(1, 10, (N, N)).astype(np.float32)
    c = np.zeros_like(a, dtype=np.float32)

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

def matrix_mult2_float(N=2000):
    print('Matrix multiplication on GPU ver.2')
    context, prog, queue = cl_boilerplate()
    # a = np.arange(1, 26, dtype=float).reshape(5, 5)
    # b = np.arange(1, 51, 2, dtype=float).reshape(5, 5)
    a = np.random.randint(1, 10, (N, N)).astype(np.float32)
    b = np.random.randint(1, 10, (N, N)).astype(np.float32)
    c = np.zeros_like(a, dtype=np.float32)

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


def matrix_mult3_float(N=2000):
    print('Matrix multiplication on GPU ver.2')
    context, prog, queue = cl_boilerplate()
    # a = np.arange(1, 26, dtype=float).reshape(5, 5)
    # b = np.arange(1, 51, 2, dtype=float).reshape(5, 5)
    a = np.random.randint(1, 10, (N, N)).astype(np.float32)
    b = np.random.randint(1, 10, (N, N)).astype(np.float32)
    c = np.zeros_like(a, dtype=np.float32)

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


if __name__ == '__main__':
    N = 2500
    c = matrix_mult_cpu(N)
    c1 = matrix_mult1(N)
    c2 = matrix_mult1_float(N)
    c3 = matrix_mult2_float(N)
    c4 = matrix_mult3_float(N)
