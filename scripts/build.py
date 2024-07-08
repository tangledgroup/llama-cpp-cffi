from cffi import FFI

ffi = FFI()

# Read the header file to define the C functions and types
with open("src/mymodule.h") as header:
    ffi.cdef(header.read())

ffi.set_source(
    "myproject._mymodule",
    """#include "mymodule.h" """,
    sources=["src/mymodule.c"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)