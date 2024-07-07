# build_cffi.py
from cffi import FFI

ffibuilder = FFI()

# Define the function signature
ffibuilder.cdef("""
    int llama_cli_main(int argc, char ** argv);
""")

# Provide the source code for the library
ffibuilder.set_source("_llama_cli",
"""
    #include "llama_cpp_cffi.h"
""",
    libraries=["./libllama-cli.so"]
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
