// include/gpuComplexSolver.h
#ifndef GPU_COMPLEX_SOLVER_H
#define GPU_COMPLEX_SOLVER_H

#include <vector>
#include <complex>

// 函数声明：使用cuSOLVER求解复数稀疏线性方程组 A x = b
bool gpu_complex_solver_cusolver(int N,
                                 const std::vector<int> &csrRowPtr,
                                 const std::vector<int> &csrColInd,
                                 const std::vector<std::complex<double>> &csrVal,
                                 const std::vector<std::complex<double>> &b,
                                 std::vector<std::complex<double>> &x,
                                 double tol = 1e-12);

#endif // GPU_COMPLEX_SOLVER_H