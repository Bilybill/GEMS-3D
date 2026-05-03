from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='mt_forward_cuda',
    ext_modules=[
        CUDAExtension('mt_forward_cuda', [
            'forward_module.cpp', 
            'src/mesh3d.cpp',
            'src/femAssemble3d.cpp',
            'src/boundaryConditions3d.cpp',
            'src/gpuComplexSolver.cu',
            'src/gpuIterativeSolver.cu',
            'src/gpuAssemble3d.cu',
            'src/gpuBoundaryPostprocess.cu',
        ],
        extra_compile_args={'cxx': ['-O3', '-std=c++17'], 'nvcc': ['-O3', '--use_fast_math', '-std=c++17']},
        libraries=['cusolver', 'cusparse', 'cublas', 'cudart'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
