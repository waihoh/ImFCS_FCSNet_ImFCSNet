import math
from Configurations import CNN
from numba import cuda, float32, uint32, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
from inspect import signature

'''
NOTE:
Using numba cuda jit wrapper for generation of simulation data in parallel.
In compile decorator, we have to define separate wrapper for different length of arguments. 
'''


####################################################################
# Poisson random numbers
####################################################################
# Return a Poisson distributed int32 and advance ``states[index]``.
# The return value is drawn from a Poisson distribution of mean=lam using
# Donald Knuth Poisson process method. Use only for small lam
@cuda.jit(device=True)
def xoroshiro128p_poisson_mult_uint32(states, index, lam):
    enlam = math.exp(-float32(lam))
    index = int64(index)
    x = uint32(0)
    prod = float32(1.0)
    while True:
        prod *= xoroshiro128p_uniform_float32(states, index)
        if prod > enlam:
            x += uint32(1)
        else:
            return x


# Return a Poisson distributed int32 and advance ``states[index]``.
# The return value is drawn from a Poisson distribution of mean=lam using
# the method of W. Hoermann. Use for moderate to large lam
@cuda.jit(device=True)
def xoroshiro128p_poisson_ptrs_uint32(states, index, lam):
    lam = float32(lam)
    index = int64(index)
    slam = math.sqrt(lam)
    loglam = math.log(lam)
    b = float32(0.931) + float32(2.53) * slam
    a = float32(-0.059) + float32(0.02483) * b
    invalpha = float32(1.1239) + float32(1.1328) / (b - float32(3.4))
    vr = float32(0.9277) - float32(3.6224) / (b - float32(2.0))
    while True:
        u = xoroshiro128p_uniform_float32(states, index) - float32(0.5)
        v = float32(1.0) - xoroshiro128p_uniform_float32(states, index)
        us = float32(0.5) - math.fabs(u)
        if us < float32(0.013) and v > us:
            continue
        fk = math.floor((float32(2.0) * a / us + b) * u + lam + float32(0.43))
        if (us >= float32(0.07)) and (v <= vr):
            return uint32(fk)
        if fk < 0.0:
            continue
        if (math.log(v) + math.log(invalpha) - math.log(a / (us * us) + b) <=
                -lam + fk * loglam - math.lgamma(fk + float32(1.0))):
            return uint32(fk)


# Return a Poisson distributed int32 and advance ``states[index]``.
# The return value is drawn from a Poisson distribution of mean=lam
@cuda.jit(device=True)
def xoroshiro128p_poisson_uint32(states, index, lam):
    if lam > 10.0:
        return xoroshiro128p_poisson_ptrs_uint32(states, index, lam)
    if lam == 0.0:
        return uint32(0)
    return xoroshiro128p_poisson_mult_uint32(states, index, lam)


####################################################################
# Generators
####################################################################
randu = xoroshiro128p_uniform_float32
randn = xoroshiro128p_normal_float32
randp = xoroshiro128p_poisson_uint32
absidx = cuda.jit(device=True)(lambda fromidx, idx: fromidx + idx)

####################################################################
# Function Decorator
####################################################################
def compile(max):
    rng = create_xoroshiro128p_states(n=max, seed=CNN.GLOBALSEED)

    # in both cases, 'rng.shape[0]' equals 'max'
    def decorator(func):
        args = len(signature(func).parameters)
        func = cuda.jit(device=True)(func)

        if args == 5:
            def kernel(rng, fromidx, toidx, d_ary1):
                idx = cuda.grid(1)
                if idx < toidx - fromidx: func(rng, fromidx, toidx, idx, d_ary1)

            kernel = cuda.jit(kernel)

            def wrapper(fromidx, toidx, ary1):
                if fromidx < 0 or toidx > max or toidx > ary1.shape[0]:
                    raise ValueError("index out of bound")
                d_ary1 = cuda.to_device(ary1[fromidx:toidx])
                blocks = (toidx - fromidx + CNN.BATCH_SIZE - 1) // CNN.BATCH_SIZE
                threads = CNN.BATCH_SIZE
                kernel[blocks, threads](rng, fromidx, toidx, d_ary1)
                cuda.synchronize()
                d_ary1.copy_to_host(ary1[fromidx:toidx])

        elif args == 6:
            def kernel(rng, fromidx, toidx, d_ary1, d_ary2):
                idx = cuda.grid(1)
                if idx < toidx - fromidx: func(rng, fromidx, toidx, idx, d_ary1, d_ary2)

            kernel = cuda.jit(kernel)

            def wrapper(fromidx, toidx, ary1, ary2):
                if fromidx < 0 or toidx > max or toidx > ary1.shape[0] or toidx > ary2.shape[0]:
                    raise ValueError("index out of bound")
                d_ary1 = cuda.to_device(ary1[fromidx:toidx])
                d_ary2 = cuda.to_device(ary2[fromidx:toidx])
                blocks = (toidx - fromidx + CNN.BATCH_SIZE - 1) // CNN.BATCH_SIZE
                threads = CNN.BATCH_SIZE
                kernel[blocks, threads](rng, fromidx, toidx, d_ary1, d_ary2)
                cuda.synchronize()
                d_ary1.copy_to_host(ary1[fromidx:toidx])
                d_ary2.copy_to_host(ary2[fromidx:toidx])

        elif args == 7:
            def kernel(rng, fromidx, toidx, d_ary1, d_ary2, d_ary3):
                idx = cuda.grid(1)
                if idx < toidx - fromidx: func(rng, fromidx, toidx, idx, d_ary1, d_ary2, d_ary3)

            kernel = cuda.jit(kernel)

            def wrapper(fromidx, toidx, ary1, ary2, ary3):
                if fromidx < 0 or toidx > max or toidx > ary1.shape[0] or toidx > ary2.shape[0] or toidx > \
                        ary3.shape[0]:
                    raise ValueError("index out of bound")
                d_ary1 = cuda.to_device(ary1[fromidx:toidx])
                d_ary2 = cuda.to_device(ary2[fromidx:toidx])
                d_ary3 = cuda.to_device(ary3[fromidx:toidx])
                blocks = (toidx - fromidx + CNN.BATCH_SIZE - 1) // CNN.BATCH_SIZE
                threads = CNN.BATCH_SIZE
                kernel[blocks, threads](rng, fromidx, toidx, d_ary1, d_ary2, d_ary3)
                cuda.synchronize()
                d_ary1.copy_to_host(ary1[fromidx:toidx])
                d_ary2.copy_to_host(ary2[fromidx:toidx])
                d_ary3.copy_to_host(ary3[fromidx:toidx])

        elif args == 8:
            def kernel(rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4):
                idx = cuda.grid(1)
                if idx < toidx - fromidx: func(rng, fromidx, toidx, idx, d_ary1, d_ary2, d_ary3, d_ary4)

            kernel = cuda.jit(kernel)

            def wrapper(fromidx, toidx, ary1, ary2, ary3, ary4):
                if (fromidx < 0 or toidx > max or toidx > ary1.shape[0] or
                        toidx > ary2.shape[0] or toidx > ary3.shape[0] or toidx > ary4.shape[0]):
                    raise ValueError("index out of bound")
                d_ary1 = cuda.to_device(ary1[fromidx:toidx])
                d_ary2 = cuda.to_device(ary2[fromidx:toidx])
                d_ary3 = cuda.to_device(ary3[fromidx:toidx])
                d_ary4 = cuda.to_device(ary4[fromidx:toidx])
                blocks = (toidx - fromidx + CNN.BATCH_SIZE - 1) // CNN.BATCH_SIZE
                threads = CNN.BATCH_SIZE
                kernel[blocks, threads](rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4)
                cuda.synchronize()
                d_ary1.copy_to_host(ary1[fromidx:toidx])
                d_ary2.copy_to_host(ary2[fromidx:toidx])
                d_ary3.copy_to_host(ary3[fromidx:toidx])
                d_ary4.copy_to_host(ary4[fromidx:toidx])

        elif args == 9:
            def kernel(rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4, d_ary5):
                idx = cuda.grid(1)
                if idx < toidx - fromidx: func(rng, fromidx, toidx, idx, d_ary1, d_ary2, d_ary3, d_ary4,
                                                d_ary5)

            kernel = cuda.jit(kernel)

            def wrapper(fromidx, toidx, ary1, ary2, ary3, ary4, ary5):
                if (fromidx < 0 or toidx > max or toidx > ary1.shape[0] or
                        toidx > ary2.shape[0] or toidx > ary3.shape[0] or toidx > ary4.shape[0] or toidx >
                        ary5.shape[0]):
                    raise ValueError("index out of bound")
                d_ary1 = cuda.to_device(ary1[fromidx:toidx])
                d_ary2 = cuda.to_device(ary2[fromidx:toidx])
                d_ary3 = cuda.to_device(ary3[fromidx:toidx])
                d_ary4 = cuda.to_device(ary4[fromidx:toidx])
                d_ary5 = cuda.to_device(ary5[fromidx:toidx])
                blocks = (toidx - fromidx + CNN.BATCH_SIZE - 1) // CNN.BATCH_SIZE
                threads = CNN.BATCH_SIZE
                kernel[blocks, threads](rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4, d_ary5)
                cuda.synchronize()
                d_ary1.copy_to_host(ary1[fromidx:toidx])
                d_ary2.copy_to_host(ary2[fromidx:toidx])
                d_ary3.copy_to_host(ary3[fromidx:toidx])
                d_ary4.copy_to_host(ary4[fromidx:toidx])
                d_ary5.copy_to_host(ary5[fromidx:toidx])

        elif args == 10:
            def kernel(rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4, d_ary5, d_ary6):
                idx = cuda.grid(1)
                if idx < toidx - fromidx: func(rng, fromidx, toidx, idx, d_ary1, d_ary2, d_ary3, d_ary4,
                                                d_ary5, d_ary6)

            kernel = cuda.jit(kernel)

            def wrapper(fromidx, toidx, ary1, ary2, ary3, ary4, ary5, ary6):
                if (fromidx < 0 or toidx > max or toidx > ary1.shape[0] or
                        toidx > ary2.shape[0] or toidx > ary3.shape[0] or toidx > ary4.shape[0] or toidx >
                        ary5.shape[0] or toidx > ary6.shape[0]):
                    raise ValueError("index out of bound")
                d_ary1 = cuda.to_device(ary1[fromidx:toidx])
                d_ary2 = cuda.to_device(ary2[fromidx:toidx])
                d_ary3 = cuda.to_device(ary3[fromidx:toidx])
                d_ary4 = cuda.to_device(ary4[fromidx:toidx])
                d_ary5 = cuda.to_device(ary5[fromidx:toidx])
                d_ary6 = cuda.to_device(ary6[fromidx:toidx])
                blocks = (toidx - fromidx + CNN.BATCH_SIZE - 1) // CNN.BATCH_SIZE
                threads = CNN.BATCH_SIZE
                kernel[blocks, threads](rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4, d_ary5, d_ary6)
                cuda.synchronize()
                d_ary1.copy_to_host(ary1[fromidx:toidx])
                d_ary2.copy_to_host(ary2[fromidx:toidx])
                d_ary3.copy_to_host(ary3[fromidx:toidx])
                d_ary4.copy_to_host(ary4[fromidx:toidx])
                d_ary5.copy_to_host(ary5[fromidx:toidx])
                d_ary6.copy_to_host(ary6[fromidx:toidx])

        else:
            raise ValueError("Number of arguments in @compile not supported")
        return wrapper

    return decorator
