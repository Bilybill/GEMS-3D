// gpuComplexSolver.cu
#include "../include/gpuComplexSolver.h"
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse_v2.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>
#include <complex>

// 辅助函数：将 std::complex<double> 转换为 cuDoubleComplex
inline cuDoubleComplex to_cuDoubleComplex(const std::complex<double>& c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}
// inline cuDoubleComplex to_cuDoubleComplex(const std::complex<double>& c) {
//     return make_cuDoubleComplex(static_cast<float>(c.real()), static_cast<float>(c.imag()));
// }

bool gpu_complex_solver_cusolver(int N,
                                 const std::vector<int> &h_csrRowPtr,
                                 const std::vector<int> &h_csrColInd,
                                 const std::vector<std::complex<double>> &h_csrVal,
                                 const std::vector<std::complex<double>> &h_b,
                                 std::vector<std::complex<double>> &x,
                                 double tol)
{
    cusolverSpHandle_t cusolverHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    cusparseMatDescr_t descrA = NULL;
    // csrlsvqrInfo_t info = NULL;

    // 创建 cuSOLVER 和 cuSPARSE 句柄
    cusolverSpCreate(&cusolverHandle);
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // 创建 csrlsvqrInfo_t 对象
    // cusolverSpCreateCsrlsvqrInfo(&info);

    int nnz = static_cast<int>(h_csrVal.size());

    // 分配 GPU 内存
    int *d_csrRowPtr = nullptr, *d_csrColInd = nullptr;
    cuDoubleComplex *d_csrVal = nullptr, *d_b = nullptr, *d_x = nullptr;

    cudaError_t cudaStat;
    cudaStat = cudaMalloc((void**)&d_csrRowPtr, (N + 1) * sizeof(int));
    if (cudaStat != cudaSuccess) { std::cerr << "cudaMalloc failed for d_csrRowPtr\n"; return false; }
    cudaStat = cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int));
    if (cudaStat != cudaSuccess) { std::cerr << "cudaMalloc failed for d_csrColInd\n"; return false; }
    cudaStat = cudaMalloc((void**)&d_csrVal, nnz * sizeof(cuDoubleComplex));
    if (cudaStat != cudaSuccess) { std::cerr << "cudaMalloc failed for d_csrVal\n"; return false; }
    cudaStat = cudaMalloc((void**)&d_b, N * sizeof(cuDoubleComplex));
    if (cudaStat != cudaSuccess) { std::cerr << "cudaMalloc failed for d_b\n"; return false; }
    cudaStat = cudaMalloc((void**)&d_x, N * sizeof(cuDoubleComplex));
    if (cudaStat != cudaSuccess) { std::cerr << "cudaMalloc failed for d_x\n"; return false; }

    // 拷贝数据到 GPU
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);

    // 将 csrVal 和 b 转换为 cuDoubleComplex
    std::vector<cuDoubleComplex> c_csrVal(nnz);
    for (int i = 0; i < nnz; i++) {
        c_csrVal[i] = to_cuDoubleComplex(h_csrVal[i]);
    }
    std::vector<cuDoubleComplex> c_b(N);
    for (int i = 0; i < N; i++) {
        c_b[i] = to_cuDoubleComplex(h_b[i]);
    }

    cudaMemcpy(d_csrVal, c_csrVal.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, c_b.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // 初始化 x 为零
    cudaMemset(d_x, 0, N * sizeof(cuDoubleComplex));

    // 使用 cusolverSpZcsrlsvqr_bufferSize 获取工作空间大小
    // size_t bufferSize = 0;
    // cusolverSpZcsrlsvqr_bufferSize(
    //     cusolverHandle,
    //     N,
    //     nnz,
    //     descrA,
    //     d_csrVal,
    //     d_csrRowPtr,
    //     d_csrColInd,
    //     d_b,
    //     tol,
    //     &bufferSize,
    //     info
    // );

    // // 分配工作空间
    // void* workspace = nullptr;
    // cudaMalloc(&workspace, bufferSize);

    // 使用 cusolverSpZcsrlsvqr 求解 A x = b
    int singularity = 0;

    cusolverStatus_t status = cusolverSpZcsrlsvqr(
        cusolverHandle,
        N,
        nnz,
        descrA,
        d_csrVal,
        d_csrRowPtr,
        d_csrColInd,
        d_b,
        tol,
        0,
        d_x,
        &singularity
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverSpZcsrlsvqr failed with status " << status << std::endl;
        // 释放资源
        cudaFree(d_csrRowPtr);
        cudaFree(d_csrColInd);
        cudaFree(d_csrVal);
        cudaFree(d_b);
        cudaFree(d_x);
        cusparseDestroyMatDescr(descrA);
        cusparseDestroy(cusparseHandle);
        cusolverSpDestroy(cusolverHandle);
        return false;
    }

    if (singularity >= 0) {
        std::cerr << "Matrix is singular at row " << singularity << std::endl;
        // 释放资源
        cudaFree(d_csrRowPtr);
        cudaFree(d_csrColInd);
        cudaFree(d_csrVal);
        cudaFree(d_b);
        cudaFree(d_x);
        cusparseDestroyMatDescr(descrA);
        cusparseDestroy(cusparseHandle);
        cusolverSpDestroy(cusolverHandle);
        return false;
    }

    // 拷贝解回主机
    std::vector<cuDoubleComplex> c_x(N);
    cudaMemcpy(c_x.data(), d_x, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    x.resize(N, std::complex<double>(0.0, 0.0));
    for (int i = 0; i < N; i++) {
        x[i] = std::complex<double>(cuCreal(c_x[i]), cuCimag(c_x[i]));
    }

    // 释放资源
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_b);
    cudaFree(d_x);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(cusparseHandle);
    cusolverSpDestroy(cusolverHandle);

    return true;
}