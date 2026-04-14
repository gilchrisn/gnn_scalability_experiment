"""
Build script for the C++ MPRW kernel PyTorch extension.

Usage:
    python setup_mprw.py build_ext --inplace

This compiles csrc/mprw_kernel.cpp into a shared library that can be
imported as ``import mprw_cpp``.  The extension is single-threaded by design
(no OpenMP), so we explicitly disable OpenMP flags.
"""

import os
import sys
from pathlib import Path

from torch.utils.cpp_extension import BuildExtension, CppExtension

from setuptools import setup

# Force single-thread: strip OpenMP from compiler flags.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Extra compile flags per platform.
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17"]
else:
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-fno-openmp",          # no OpenMP
        "-march=native",        # use host instruction set
        "-ffast-math",          # allow FP reordering (only RNG, no precision issue)
    ]

setup(
    name="mprw_cpp",
    version="0.1.0",
    description="Cache-optimised C++ MPRW materialisation kernel",
    ext_modules=[
        CppExtension(
            name="mprw_cpp",
            sources=[str(Path("csrc") / "mprw_kernel.cpp")],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
