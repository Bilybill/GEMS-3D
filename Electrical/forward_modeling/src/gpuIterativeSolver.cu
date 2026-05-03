// src/gpuIterativeSolver.cu
#include "../include/gpuIterativeSolver.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <iostream>
#include <cmath>
#include <stdexcept>

struct GpuBiCGStabContext {
    int N = 0;
    int nnz = 0;
    bool structureInitialized = false;
    std::vector<int> diagonalIndices;
    cublasHandle_t cublasH = nullptr;
    cusparseHandle_t cusparseH = nullptr;
    int *d_csrRowPtr = nullptr;
    int *d_csrColInd = nullptr;
    int *d_diagonalIndices = nullptr;
    cuDoubleComplex *d_csrVal = nullptr;
    cuDoubleComplex *d_b = nullptr;
    cuDoubleComplex *d_x = nullptr;
    cuDoubleComplex *d_Minv = nullptr;
    cuDoubleComplex *d_r = nullptr;
    cuDoubleComplex *d_r0_hat = nullptr;
    cuDoubleComplex *d_p = nullptr;
    cuDoubleComplex *d_v = nullptr;
    cuDoubleComplex *d_s = nullptr;
    cuDoubleComplex *d_t = nullptr;
    cuDoubleComplex *d_yp = nullptr;
    cuDoubleComplex *d_ys = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    void* dBuffer = nullptr;
    size_t bufferSize = 0;
};

__global__ void complexElementWiseMul(int n, const cuDoubleComplex* a, const cuDoubleComplex* b, cuDoubleComplex* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = cuCmul(a[idx], b[idx]);
    }
}

__global__ void computeJacobiInverseKernel(int n,
                                           const int* diagonalIndices,
                                           const cuDoubleComplex* csrVal,
                                           cuDoubleComplex* Minv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int diagIndex = diagonalIndices[idx];
    cuDoubleComplex diag = make_cuDoubleComplex(1.0, 0.0);
    if (diagIndex >= 0) {
        diag = csrVal[diagIndex];
    }

    double mag_sq = cuCreal(diag) * cuCreal(diag) + cuCimag(diag) * cuCimag(diag);
    if (mag_sq > 1e-30) {
        Minv[idx] = make_cuDoubleComplex(cuCreal(diag) / mag_sq, -cuCimag(diag) / mag_sq);
    } else {
        Minv[idx] = make_cuDoubleComplex(1.0, 0.0);
    }
}

inline cuDoubleComplex to_cuDoubleComplex(const std::complex<double>& c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}

namespace {

GpuBiCGStabContext* allocate_context(int N, int nnz) {
    auto* context = new GpuBiCGStabContext();
    context->N = N;
    context->nnz = nnz;

    cublasCreate(&context->cublasH);
    cusparseCreate(&context->cusparseH);

    cudaMalloc((void**)&context->d_csrRowPtr, (N + 1) * sizeof(int));
    cudaMalloc((void**)&context->d_csrColInd, nnz * sizeof(int));
    cudaMalloc((void**)&context->d_diagonalIndices, N * sizeof(int));
    cudaMalloc((void**)&context->d_csrVal, nnz * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_b, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_x, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_Minv, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_r, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_r0_hat, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_p, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_v, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_s, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_t, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_yp, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&context->d_ys, N * sizeof(cuDoubleComplex));

    cusparseCreateCsr(&context->matA, N, N, nnz, context->d_csrRowPtr, context->d_csrColInd, context->d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

    cuDoubleComplex alpha_c = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta_c  = make_cuDoubleComplex(0.0, 0.0);
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, N, context->d_p, CUDA_C_64F);
    cusparseCreateDnVec(&vecY, N, context->d_t, CUDA_C_64F);
    cusparseSpMV_bufferSize(context->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_c, context->matA, vecX, &beta_c, vecY,
                            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &context->bufferSize);
    cudaMalloc(&context->dBuffer, context->bufferSize);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return context;
}

void free_context(GpuBiCGStabContext* context) {
    if (!context) {
        return;
    }

    cudaFree(context->d_csrRowPtr);
    cudaFree(context->d_csrColInd);
    cudaFree(context->d_diagonalIndices);
    cudaFree(context->d_csrVal);
    cudaFree(context->d_b);
    cudaFree(context->d_x);
    cudaFree(context->d_Minv);
    cudaFree(context->d_r);
    cudaFree(context->d_r0_hat);
    cudaFree(context->d_p);
    cudaFree(context->d_v);
    cudaFree(context->d_s);
    cudaFree(context->d_t);
    cudaFree(context->d_yp);
    cudaFree(context->d_ys);
    cudaFree(context->dBuffer);

    if (context->matA) {
        cusparseDestroySpMat(context->matA);
    }
    if (context->cusparseH) {
        cusparseDestroy(context->cusparseH);
    }
    if (context->cublasH) {
        cublasDestroy(context->cublasH);
    }
    delete context;
}

void validate_context(const GpuBiCGStabContext* context, int N, int nnz) {
    if (!context) {
        return;
    }
    if (context->N != N || context->nnz != nnz) {
        throw std::runtime_error("GpuBiCGStabContext size mismatch");
    }
}

void initialize_structure_if_needed(GpuBiCGStabContext* context,
                                    const std::vector<int> &h_csrRowPtr,
                                    const std::vector<int> &h_csrColInd)
{
    if (context->structureInitialized) {
        return;
    }

    cudaMemcpy(context->d_csrRowPtr, h_csrRowPtr.data(), (context->N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(context->d_csrColInd, h_csrColInd.data(), context->nnz * sizeof(int), cudaMemcpyHostToDevice);

    context->diagonalIndices.assign(context->N, -1);
    for (int row = 0; row < context->N; ++row) {
        for (int idx = h_csrRowPtr[row]; idx < h_csrRowPtr[row + 1]; ++idx) {
            if (h_csrColInd[idx] == row) {
                context->diagonalIndices[row] = idx;
                break;
            }
        }
    }
    cudaMemcpy(context->d_diagonalIndices, context->diagonalIndices.data(),
               context->N * sizeof(int), cudaMemcpyHostToDevice);
    context->structureInitialized = true;
}

bool solve_bicgstab_in_context(GpuBiCGStabContext* context,
                               double tol,
                               int maxIter,
                               SolverStats* stats)
{
    if (stats) {
        *stats = SolverStats{};
    }

    int N = context->N;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    computeJacobiInverseKernel<<<blocks, threads>>>(N, context->d_diagonalIndices, context->d_csrVal, context->d_Minv);

    cuDoubleComplex alpha_c = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta_c  = make_cuDoubleComplex(0.0, 0.0);

    auto SpMV_A = [&](cuDoubleComplex* in, cuDoubleComplex* out) {
        cusparseDnVecDescr_t vIn, vOut;
        cusparseCreateDnVec(&vIn, N, in, CUDA_C_64F);
        cusparseCreateDnVec(&vOut, N, out, CUDA_C_64F);
        cusparseSpMV(context->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_c, context->matA, vIn, &beta_c, vOut,
                     CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, context->dBuffer);
        cusparseDestroyDnVec(vIn);
        cusparseDestroyDnVec(vOut);
    };

    double norm_b, res_norm;
    cublasDznrm2(context->cublasH, N, context->d_b, 1, &norm_b);
    if (norm_b == 0.0) {
        norm_b = 1.0;
    }

    cuDoubleComplex rho_prev = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex omega = make_cuDoubleComplex(1.0, 0.0);

    cudaMemset(context->d_p, 0, N * sizeof(cuDoubleComplex));
    cudaMemset(context->d_v, 0, N * sizeof(cuDoubleComplex));

    cudaMemcpy(context->d_r, context->d_b, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    cublasDznrm2(context->cublasH, N, context->d_x, 1, &res_norm);
    if (res_norm > 0.0) {
        SpMV_A(context->d_x, context->d_t);
        cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);
        cublasZaxpy(context->cublasH, N, &neg_one, context->d_t, 1, context->d_r, 1);
    }

    cudaMemcpy(context->d_r0_hat, context->d_r, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

    cublasDznrm2(context->cublasH, N, context->d_r, 1, &res_norm);
    if (res_norm / norm_b < tol) {
        std::cout << "  -> BiCGStab converged immediately (Warm Start). Final res: " << res_norm / norm_b << std::endl;
        if (stats) {
            stats->iterations = 0;
            stats->finalResidual = res_norm / norm_b;
            stats->converged = true;
        }
        return true;
    }

    int iter = 0;
    bool converged = false;

    for (iter = 1; iter <= maxIter; ++iter) {
        cuDoubleComplex rho;
        cublasZdotu(context->cublasH, N, context->d_r0_hat, 1, context->d_r, 1, &rho);

        if (cuCabs(rho) < 1e-30) {
            break;
        }

        cuDoubleComplex beta = cuCmul(cuCdiv(rho, rho_prev), cuCdiv(alpha, omega));

        cuDoubleComplex neg_omega = make_cuDoubleComplex(-cuCreal(omega), -cuCimag(omega));
        cublasZaxpy(context->cublasH, N, &neg_omega, context->d_v, 1, context->d_p, 1);
        cublasZscal(context->cublasH, N, &beta, context->d_p, 1);
        cuDoubleComplex one_c = make_cuDoubleComplex(1.0, 0.0);
        cublasZaxpy(context->cublasH, N, &one_c, context->d_r, 1, context->d_p, 1);

        complexElementWiseMul<<<blocks, threads>>>(N, context->d_Minv, context->d_p, context->d_yp);
        SpMV_A(context->d_yp, context->d_v);

        cuDoubleComplex r0_v;
        cublasZdotu(context->cublasH, N, context->d_r0_hat, 1, context->d_v, 1, &r0_v);
        if (cuCabs(r0_v) < 1e-30) {
            break;
        }
        alpha = cuCdiv(rho, r0_v);

        cudaMemcpy(context->d_s, context->d_r, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-cuCreal(alpha), -cuCimag(alpha));
        cublasZaxpy(context->cublasH, N, &neg_alpha, context->d_v, 1, context->d_s, 1);

        cublasDznrm2(context->cublasH, N, context->d_s, 1, &res_norm);
        if (res_norm / norm_b < tol) {
            cublasZaxpy(context->cublasH, N, &alpha, context->d_yp, 1, context->d_x, 1);
            converged = true;
            break;
        }

        complexElementWiseMul<<<blocks, threads>>>(N, context->d_Minv, context->d_s, context->d_ys);
        SpMV_A(context->d_ys, context->d_t);

        cuDoubleComplex ts, tt;
        cublasZdotc(context->cublasH, N, context->d_t, 1, context->d_s, 1, &ts);
        cublasZdotc(context->cublasH, N, context->d_t, 1, context->d_t, 1, &tt);
        if (cuCabs(tt) < 1e-30) {
            break;
        }
        omega = cuCdiv(ts, tt);

        cublasZaxpy(context->cublasH, N, &alpha, context->d_yp, 1, context->d_x, 1);
        cublasZaxpy(context->cublasH, N, &omega, context->d_ys, 1, context->d_x, 1);

        cudaMemcpy(context->d_r, context->d_s, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        neg_omega = make_cuDoubleComplex(-cuCreal(omega), -cuCimag(omega));
        cublasZaxpy(context->cublasH, N, &neg_omega, context->d_t, 1, context->d_r, 1);

        rho_prev = rho;

        cublasDznrm2(context->cublasH, N, context->d_r, 1, &res_norm);
        if (res_norm / norm_b < tol) {
            converged = true;
            break;
        }

        if (iter % 500 == 0) {
            std::cout << "  [Jacobi-BiCGStab] Iter " << iter << " - Residual: " << res_norm / norm_b << std::endl;
        }
    }

    if (converged) {
        std::cout << "  -> BiCGStab converged in " << iter << " iterations. Final res: " << res_norm / norm_b << std::endl;
    } else {
        std::cerr << "  -> BiCGStab failed to converge within maxIter. Final res: " << res_norm / norm_b << std::endl;
    }

    if (stats) {
        stats->iterations = iter;
        stats->finalResidual = res_norm / norm_b;
        stats->converged = converged;
    }

    return converged;
}

} // namespace

GpuBiCGStabContext* create_gpu_bicgstab_context(int N, int nnz)
{
    return allocate_context(N, nnz);
}

void destroy_gpu_bicgstab_context(GpuBiCGStabContext* context)
{
    free_context(context);
}

void initialize_gpu_bicgstab_structure(GpuBiCGStabContext* context,
                                       const std::vector<int> &h_csrRowPtr,
                                       const std::vector<int> &h_csrColInd)
{
    validate_context(context, static_cast<int>(h_csrRowPtr.size()) - 1, static_cast<int>(h_csrColInd.size()));
    initialize_structure_if_needed(context, h_csrRowPtr, h_csrColInd);
}

const int* get_gpu_bicgstab_row_ptr_device(const GpuBiCGStabContext* context)
{
    return context->d_csrRowPtr;
}

const int* get_gpu_bicgstab_col_ind_device(const GpuBiCGStabContext* context)
{
    return context->d_csrColInd;
}

cuDoubleComplex* get_gpu_bicgstab_matrix_values_device(GpuBiCGStabContext* context)
{
    return context->d_csrVal;
}

cuDoubleComplex* get_gpu_bicgstab_rhs_device(GpuBiCGStabContext* context)
{
    return context->d_b;
}

bool gpu_complex_solver_bicgstab_device(GpuBiCGStabContext* context,
                                        cuDoubleComplex* d_x_io,
                                        double tol,
                                        int maxIter,
                                        SolverStats* stats)
{
    if (!context) {
        throw std::runtime_error("GpuBiCGStabContext is required for device solve");
    }
    if (!context->structureInitialized) {
        throw std::runtime_error("GpuBiCGStabContext structure must be initialized before device solve");
    }

    if (d_x_io != context->d_x) {
        cudaMemcpy(context->d_x, d_x_io, context->N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }
    bool ok = solve_bicgstab_in_context(context, tol, maxIter, stats);
    if (d_x_io != context->d_x) {
        cudaMemcpy(d_x_io, context->d_x, context->N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }
    return ok;
}

bool gpu_complex_solver_bicgstab(int N,
                                 const std::vector<int> &h_csrRowPtr,
                                 const std::vector<int> &h_csrColInd,
                                 const std::vector<std::complex<double>> &h_csrVal,
                                 const std::vector<std::complex<double>> &h_b,
                                 std::vector<std::complex<double>> &x,
                                 double tol,
                                 int maxIter,
                                 GpuBiCGStabContext* context,
                                 SolverStats* stats)
{
    int nnz = static_cast<int>(h_csrVal.size());
    validate_context(context, N, nnz);
    bool ownsContext = false;
    if (!context) {
        context = allocate_context(N, nnz);
        ownsContext = true;
    }

    if (stats) {
        *stats = SolverStats{};
    }

    initialize_structure_if_needed(context, h_csrRowPtr, h_csrColInd);

    std::vector<cuDoubleComplex> c_csrVal(nnz), c_b(N), c_x(N);
    for (int i = 0; i < nnz; i++) c_csrVal[i] = to_cuDoubleComplex(h_csrVal[i]);
    for (int i = 0; i < N; i++) {
        c_b[i] = to_cuDoubleComplex(h_b[i]);
        c_x[i] = to_cuDoubleComplex(x[i]);
    }

    cudaMemcpy(context->d_csrVal, c_csrVal.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(context->d_b, c_b.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(context->d_x, c_x.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    bool converged = solve_bicgstab_in_context(context, tol, maxIter, stats);

    cudaMemcpy(c_x.data(), context->d_x, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        x[i] = std::complex<double>(cuCreal(c_x[i]), cuCimag(c_x[i]));
    }

    if (ownsContext) {
        free_context(context);
    }

    return converged;
}
