# build_cffi.py
from cffi import FFI

ffibuilder = FFI()

# Define the function signature
ffibuilder.cdef("""
    void say_hello(void);
""")

# Provide the source code for the library
ffibuilder.set_source("_example",
"""
    #include "example.h"
""",
    sources=["example.c"]
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
