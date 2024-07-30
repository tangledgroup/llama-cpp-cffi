__all__ = ['is_cuda_available']

try:
    from numba import cuda
except ImportError:
    pass


def is_cuda_available():
    r: bool = False
    
    try:
        r = cuda.is_available()
    except Exception:
        r = False

    return r
