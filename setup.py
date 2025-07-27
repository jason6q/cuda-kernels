"""
    Reference: https://github.com/pytorch/extension-cpp/blob/master/setup.py
"""
import os
from pathlib import Path

from setuptools import setup, Extension
from torch.utils import cpp_extension

LIB_NAME = "j6q_cu_ext"
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
SCRIPT_DIR = Path(__file__).parent.absolute()

def get_extensions():
    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not DEBUG_MODE else "-O0", # GCC Optimization flags
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000" # Use CPython Stable Limited API from Python.h
        ],
        "nvcc": [
            "-O3" if not DEBUG_MODE else "-O0" # NVCC Optimization flags
        ]
    }

    if DEBUG_MODE:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    sources = [str(cpp) for cpp in (SCRIPT_DIR / LIB_NAME / "csrc").glob("*.cpp")]
    cuda_sources = [str(cu) for cu in (SCRIPT_DIR / LIB_NAME / "csrc/cuda").glob("*.cu")]

    print("Found CUDA sources: ", cuda_sources)

    ext_module = [
        cpp_extension.CUDAExtension(
            f"{LIB_NAME}._C",
            sources + cuda_sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=True # https://docs.python.org/3/c-api/stable.html#limited-api-caveats
        )
    ]

    return ext_module

setup(name=LIB_NAME,
    ext_modules= get_extensions(),
    install_requires=["torch"],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}}
)
