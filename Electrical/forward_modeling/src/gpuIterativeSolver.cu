// src/gpuIterativeSolver.cu
#include "../include/gpuIterativeSolver.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <complex>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>

struct GpuBiCGStabContext {
    int N = 0;
    int nnz = 0;
    int device = -1;
    bool structureInitialized = false;
    bool preconditionerInitialized = false;
    std::vector<int> diagonalIndices;
    cublasHandle_t cublasH = nullptr;
    cusparseHandle_t cusparseH = nullptr;
    cudaStream_t stream = nullptr;
    int *d_csrRowPtr = nullptr;
    int *d_csrColInd = nullptr;
    int *d_diagonalIndices = nullptr;
    int *d_preconditionerInvalid = nullptr;
    cuDoubleComplex *d_csrVal = nullptr;
    cuDoubleComplex *d_b = nullptr;
    cuDoubleComplex *d_x = nullptr;
    cuDoubleComplex *d_Minv = nullptr;
    cuDoubleComplex *d_r = nullptr;
    cuDoubleComplex *d_r0_hat = nullptr;
    cuDoubleComplex *d_p = nullptr;
    cuDoubleComplex *d_v = nullptr;
    cuDoubleComplex *d_t = nullptr;
    cuDoubleComplex *d_ys = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecIn = nullptr;
    cusparseDnVecDescr_t vecOut = nullptr;
    void* dBuffer = nullptr;
    size_t bufferSize = 0;
};

__global__ void complexElementWiseMul(int n, const cuDoubleComplex* a, const cuDoubleComplex* b, cuDoubleComplex* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = cuCmul(a[idx], b[idx]);
    }
}

__global__ void updateSearchDirectionKernel(
    int n,
    const cuDoubleComplex* r,
    cuDoubleComplex beta,
    cuDoubleComplex omega,
    const cuDoubleComplex* v,
    cuDoubleComplex* p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p[idx] = cuCadd(r[idx], cuCmul(beta, cuCsub(p[idx], cuCmul(omega, v[idx]))));
    }
}

__global__ void updateSolutionAndResidualKernel(
    int n,
    cuDoubleComplex alpha,
    cuDoubleComplex omega,
    const cuDoubleComplex* yp,
    const cuDoubleComplex* ys,
    const cuDoubleComplex* t,
    cuDoubleComplex* x,
    cuDoubleComplex* r)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = cuCadd(x[idx], cuCadd(cuCmul(alpha, yp[idx]), cuCmul(omega, ys[idx])));
        r[idx] = cuCsub(r[idx], cuCmul(omega, t[idx]));
    }
}

__global__ void computeJacobiInverseKernel(int n,
                                           const int* diagonalIndices,
                                           const cuDoubleComplex* csrVal,
                                           cuDoubleComplex* Minv,
                                           int* invalidFlag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int diagIndex = diagonalIndices[idx];
    cuDoubleComplex diag = make_cuDoubleComplex(1.0, 0.0);
    if (diagIndex >= 0) {
        diag = csrVal[diagIndex];
    }

    double realPart = cuCreal(diag);
    double imagPart = cuCimag(diag);
    double scale = fmax(fabs(realPart), fabs(imagPart));
    cuDoubleComplex inverse = make_cuDoubleComplex(1.0, 0.0);
    bool valid = diagIndex >= 0 && isfinite(scale) && scale > 0.0;
    if (valid) {
        double scaledReal = realPart / scale;
        double scaledImag = imagPart / scale;
        double denominator = scaledReal * scaledReal + scaledImag * scaledImag;
        double inverseScale = 1.0 / scale;
        double inverseReal = scaledReal * inverseScale / denominator;
        double inverseImag = -scaledImag * inverseScale / denominator;
        if (isfinite(inverseReal) && isfinite(inverseImag)) {
            inverse = make_cuDoubleComplex(inverseReal, inverseImag);
        } else {
            valid = false;
        }
    }
    if (!valid) {
        atomicExch(invalidFlag, 1);
    }
    Minv[idx] = inverse;
}

inline cuDoubleComplex to_cuDoubleComplex(const std::complex<double>& c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}

namespace {

void checkCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

void checkCublas(cublasStatus_t status, const char* operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status "
            + std::to_string(static_cast<int>(status))
        );
    }
}

void checkCusparse(cusparseStatus_t status, const char* operation)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuSPARSE status "
            + std::to_string(static_cast<int>(status))
        );
    }
}

// Keep this host-side comparison overflow-safe when the two residual scales
// are very large.  The solver obtains these scalars through synchronous BLAS
// calls, so evaluating the long-double product here does not add device work.
bool scaledBreakdown(cuDoubleComplex value, double leftNorm, double rightNorm)
{
    if (!std::isfinite(cuCreal(value)) || !std::isfinite(cuCimag(value))
        || !std::isfinite(leftNorm) || leftNorm < 0.0
        || !std::isfinite(rightNorm) || rightNorm < 0.0) {
        return true;
    }

    const long double scale = static_cast<long double>(leftNorm)
        * static_cast<long double>(rightNorm);
    const long double magnitude = std::hypot(
        static_cast<long double>(cuCreal(value)),
        static_cast<long double>(cuCimag(value))
    );
    if (scale == 0.0L) {
        return magnitude == 0.0L;
    }
    return magnitude <= static_cast<long double>(std::numeric_limits<double>::epsilon())
        * scale;
}

void free_context(GpuBiCGStabContext* context);

GpuBiCGStabContext* allocate_context(int N, int nnz) {
    if (N <= 0 || N == std::numeric_limits<int>::max() || nnz <= 0) {
        throw std::invalid_argument("BiCGStab dimensions must be positive");
    }
    auto* context = new GpuBiCGStabContext();
    context->N = N;
    context->nnz = nnz;

    try {
    checkCuda(cudaGetDevice(&context->device), "querying BiCGStab CUDA device");
    checkCublas(cublasCreate(&context->cublasH), "cublasCreate");
    checkCusparse(cusparseCreate(&context->cusparseH), "cusparseCreate");

    checkCuda(cudaMalloc(
        (void**)&context->d_csrRowPtr,
        (static_cast<size_t>(N) + 1) * sizeof(int)
    ), "allocating CSR row pointers");
    checkCuda(cudaMalloc((void**)&context->d_csrColInd, nnz * sizeof(int)), "allocating CSR columns");
    checkCuda(cudaMalloc((void**)&context->d_diagonalIndices, N * sizeof(int)), "allocating diagonal indices");
    checkCuda(cudaMalloc((void**)&context->d_preconditionerInvalid, sizeof(int)), "allocating preconditioner validation flag");
    checkCuda(cudaMalloc((void**)&context->d_csrVal, nnz * sizeof(cuDoubleComplex)), "allocating CSR values");
    checkCuda(cudaMalloc((void**)&context->d_b, N * sizeof(cuDoubleComplex)), "allocating RHS");
    checkCuda(cudaMalloc((void**)&context->d_x, N * sizeof(cuDoubleComplex)), "allocating solution");
    checkCuda(cudaMalloc((void**)&context->d_Minv, N * sizeof(cuDoubleComplex)), "allocating Jacobi inverse");
    checkCuda(cudaMalloc((void**)&context->d_r, N * sizeof(cuDoubleComplex)), "allocating residual");
    checkCuda(cudaMalloc((void**)&context->d_r0_hat, N * sizeof(cuDoubleComplex)), "allocating shadow residual");
    checkCuda(cudaMalloc((void**)&context->d_p, N * sizeof(cuDoubleComplex)), "allocating search direction");
    checkCuda(cudaMalloc((void**)&context->d_v, N * sizeof(cuDoubleComplex)), "allocating Krylov vector");
    checkCuda(cudaMalloc((void**)&context->d_t, N * sizeof(cuDoubleComplex)), "allocating auxiliary vector");
    checkCuda(cudaMalloc((void**)&context->d_ys, N * sizeof(cuDoubleComplex)), "allocating preconditioned residual");

    checkCusparse(cusparseCreateCsr(
        &context->matA, N, N, nnz, context->d_csrRowPtr, context->d_csrColInd,
        context->d_csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F
    ), "cusparseCreateCsr");

    cuDoubleComplex alpha_c = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta_c  = make_cuDoubleComplex(0.0, 0.0);
    checkCusparse(cusparseCreateDnVec(&context->vecIn, N, context->d_p, CUDA_C_64F), "cusparseCreateDnVec input");
    checkCusparse(cusparseCreateDnVec(&context->vecOut, N, context->d_t, CUDA_C_64F), "cusparseCreateDnVec output");
    checkCusparse(cusparseSpMV_bufferSize(
        context->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_c, context->matA, context->vecIn, &beta_c, context->vecOut,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &context->bufferSize
    ), "cusparseSpMV_bufferSize");
    checkCuda(cudaMalloc(&context->dBuffer, context->bufferSize), "allocating SpMV workspace");
    } catch (...) {
        free_context(context);
        throw;
    }

    return context;
}

void free_context(GpuBiCGStabContext* context) {
    if (!context) {
        return;
    }

    int previousDevice = -1;
    cudaGetDevice(&previousDevice);
    bool switchedDevice = context->device >= 0 && previousDevice >= 0
        && context->device != previousDevice;
    if (switchedDevice) {
        cudaSetDevice(context->device);
    }

    if (context->vecIn) {
        cusparseDestroyDnVec(context->vecIn);
    }
    if (context->vecOut) {
        cusparseDestroyDnVec(context->vecOut);
    }
    if (context->matA) {
        cusparseDestroySpMat(context->matA);
    }
    if (context->cusparseH) {
        cusparseDestroy(context->cusparseH);
    }
    if (context->cublasH) {
        cublasDestroy(context->cublasH);
    }

    cudaFree(context->d_csrRowPtr);
    cudaFree(context->d_csrColInd);
    cudaFree(context->d_diagonalIndices);
    cudaFree(context->d_preconditionerInvalid);
    cudaFree(context->d_csrVal);
    cudaFree(context->d_b);
    cudaFree(context->d_x);
    cudaFree(context->d_Minv);
    cudaFree(context->d_r);
    cudaFree(context->d_r0_hat);
    cudaFree(context->d_p);
    cudaFree(context->d_v);
    cudaFree(context->d_t);
    cudaFree(context->d_ys);
    cudaFree(context->dBuffer);
    if (switchedDevice) {
        cudaSetDevice(previousDevice);
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

    if (h_csrRowPtr.size() != static_cast<size_t>(context->N + 1)
        || h_csrColInd.size() != static_cast<size_t>(context->nnz)) {
        throw std::invalid_argument("CSR arrays do not match the BiCGStab context dimensions");
    }
    if (h_csrRowPtr.front() != 0 || h_csrRowPtr.back() != context->nnz) {
        throw std::invalid_argument("CSR row pointers must start at zero and end at nnz");
    }
    for (int row = 0; row < context->N; ++row) {
        if (h_csrRowPtr[row] > h_csrRowPtr[row + 1]) {
            throw std::invalid_argument("CSR row pointers must be nondecreasing");
        }
    }
    for (int col : h_csrColInd) {
        if (col < 0 || col >= context->N) {
            throw std::out_of_range("CSR column index is outside the matrix dimensions");
        }
    }

    checkCuda(cudaMemcpy(
        context->d_csrRowPtr, h_csrRowPtr.data(),
        (context->N + 1) * sizeof(int), cudaMemcpyHostToDevice
    ), "copying CSR row pointers");
    checkCuda(cudaMemcpy(
        context->d_csrColInd, h_csrColInd.data(),
        context->nnz * sizeof(int), cudaMemcpyHostToDevice
    ), "copying CSR columns");

    context->diagonalIndices.assign(context->N, -1);
    for (int row = 0; row < context->N; ++row) {
        for (int idx = h_csrRowPtr[row]; idx < h_csrRowPtr[row + 1]; ++idx) {
            if (h_csrColInd[idx] == row) {
                context->diagonalIndices[row] = idx;
                break;
            }
        }
        if (context->diagonalIndices[row] < 0) {
            throw std::runtime_error(
                "CSR structure is missing a diagonal entry at row "
                + std::to_string(row)
            );
        }
    }
    checkCuda(cudaMemcpy(
        context->d_diagonalIndices, context->diagonalIndices.data(),
        context->N * sizeof(int), cudaMemcpyHostToDevice
    ), "copying diagonal indices");
    context->structureInitialized = true;
}

bool solve_bicgstab_in_context(GpuBiCGStabContext* context,
                               const cuDoubleComplex* d_b,
                               cuDoubleComplex* d_x,
                               bool rebuildPreconditioner,
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

    if (rebuildPreconditioner) {
        checkCuda(cudaMemsetAsync(
            context->d_preconditionerInvalid, 0, sizeof(int), context->stream
        ), "clearing preconditioner validation flag");
        computeJacobiInverseKernel<<<blocks, threads, 0, context->stream>>>(
            N, context->d_diagonalIndices, context->d_csrVal, context->d_Minv,
            context->d_preconditionerInvalid
        );
        checkCuda(cudaGetLastError(), "building Jacobi inverse");
        int preconditionerInvalid = 0;
        checkCuda(cudaMemcpyAsync(
            &preconditionerInvalid, context->d_preconditionerInvalid, sizeof(int),
            cudaMemcpyDeviceToHost, context->stream
        ), "reading preconditioner validation flag");
        checkCuda(cudaStreamSynchronize(context->stream), "validating Jacobi inverse");
        if (preconditionerInvalid != 0) {
            throw std::runtime_error(
                "BiCGStab matrix has a zero or non-finite diagonal entry"
            );
        }
        context->preconditionerInitialized = true;
    } else if (!context->preconditionerInitialized) {
        throw std::runtime_error(
            "BiCGStab preconditioner must be built before it can be reused"
        );
    }

    cuDoubleComplex alpha_c = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta_c  = make_cuDoubleComplex(0.0, 0.0);

    auto SpMV_A = [&](cuDoubleComplex* in, cuDoubleComplex* out) {
        checkCusparse(cusparseDnVecSetValues(context->vecIn, in), "cusparseDnVecSetValues input");
        checkCusparse(cusparseDnVecSetValues(context->vecOut, out), "cusparseDnVecSetValues output");
        checkCusparse(cusparseSpMV(
            context->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_c, context->matA, context->vecIn, &beta_c, context->vecOut,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, context->dBuffer
        ), "cusparseSpMV");
    };

    auto SpMV_LeftJacobi = [&](cuDoubleComplex* in, cuDoubleComplex* out) {
        SpMV_A(in, out);
        complexElementWiseMul<<<blocks, threads, 0, context->stream>>>(
            N, context->d_Minv, out, out
        );
        checkCuda(cudaGetLastError(), "left Jacobi-scaling SpMV output");
    };

    auto finiteComplex = [](cuDoubleComplex value) {
        return std::isfinite(cuCreal(value)) && std::isfinite(cuCimag(value));
    };
    double norm_b = 0.0;
    double res_norm = std::numeric_limits<double>::infinity();
    checkCublas(cublasDznrm2(context->cublasH, N, d_b, 1, &norm_b), "cublasDznrm2 RHS");
    if (!std::isfinite(norm_b)) {
        throw std::runtime_error("BiCGStab RHS norm is not finite");
    }
    // Solve the left-scaled system M^-1 A x = M^-1 b.  This makes the Krylov
    // recurrence reduce the interior-equation residual directly instead of
    // allowing unit Dirichlet rows to dominate the stopping norm.
    complexElementWiseMul<<<blocks, threads, 0, context->stream>>>(
        N, context->d_Minv, d_b, context->d_ys
    );
    checkCuda(cudaGetLastError(), "Jacobi-scaling RHS for convergence check");
    double jacobiScaledNormB = 0.0;
    checkCublas(cublasDznrm2(
        context->cublasH, N, context->d_ys, 1, &jacobiScaledNormB
    ), "cublasDznrm2 Jacobi-scaled RHS");
    if (!std::isfinite(jacobiScaledNormB)) {
        throw std::runtime_error("BiCGStab Jacobi-scaled RHS norm is not finite");
    }

    auto relativeResidual = [](double residualNorm, double rhsNorm) {
        if (rhsNorm == 0.0) {
            return residualNorm == 0.0
                ? 0.0
                : std::numeric_limits<double>::infinity();
        }
        return residualNorm / rhsNorm;
    };

    double verifiedTrueResidual = std::numeric_limits<double>::infinity();
    double verifiedScaledResidual = std::numeric_limits<double>::infinity();
    double verifiedScaledResidualNorm = std::numeric_limits<double>::infinity();
    auto verifyTrueResidual = [&]() {
        SpMV_A(d_x, context->d_t);
        checkCuda(cudaMemcpyAsync(
            context->d_ys, d_b, N * sizeof(cuDoubleComplex),
            cudaMemcpyDeviceToDevice, context->stream
        ), "copying RHS for residual");
        cuDoubleComplex negOne = make_cuDoubleComplex(-1.0, 0.0);
        checkCublas(cublasZaxpy(
            context->cublasH, N, &negOne, context->d_t, 1, context->d_ys, 1
        ), "cublasZaxpy residual");
        double trueResidualNorm = 0.0;
        checkCublas(cublasDznrm2(
            context->cublasH, N, context->d_ys, 1, &trueResidualNorm
        ), "cublasDznrm2 residual");

        complexElementWiseMul<<<blocks, threads, 0, context->stream>>>(
            N, context->d_Minv, context->d_ys, context->d_ys
        );
        checkCuda(cudaGetLastError(), "Jacobi-scaling residual");
        checkCublas(cublasDznrm2(
            context->cublasH, N, context->d_ys, 1,
            &verifiedScaledResidualNorm
        ), "cublasDznrm2 Jacobi-scaled residual");

        verifiedTrueResidual = relativeResidual(trueResidualNorm, norm_b);
        verifiedScaledResidual = relativeResidual(
            verifiedScaledResidualNorm, jacobiScaledNormB
        );
        return std::isfinite(verifiedTrueResidual)
            && std::isfinite(verifiedScaledResidual)
            && std::max(verifiedTrueResidual, verifiedScaledResidual) < tol;
    };

    // True-residual checks use d_ys as scratch so periodic verification does
    // not destroy the active BiCGStab recurrence. Replace it only when a
    // restart is actually required.
    auto replaceRecurrenceResidual = [&]() {
        if (!std::isfinite(verifiedScaledResidualNorm)) {
            throw std::runtime_error(
                "BiCGStab verified scaled residual norm is not finite"
            );
        }
        checkCuda(cudaMemcpyAsync(
            context->d_r, context->d_ys, N * sizeof(cuDoubleComplex),
            cudaMemcpyDeviceToDevice, context->stream
        ), "replacing BiCGStab recurrence residual");
        res_norm = verifiedScaledResidualNorm;
    };

    cuDoubleComplex rho_prev = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex omega = make_cuDoubleComplex(1.0, 0.0);

    checkCuda(cudaMemsetAsync(
        context->d_p, 0, N * sizeof(cuDoubleComplex), context->stream
    ), "clearing search direction");
    checkCuda(cudaMemsetAsync(
        context->d_v, 0, N * sizeof(cuDoubleComplex), context->stream
    ), "clearing Krylov vector");

    const bool initiallyConverged = verifyTrueResidual();
    if (!std::isfinite(verifiedTrueResidual)
        || !std::isfinite(verifiedScaledResidual)
        || !std::isfinite(verifiedScaledResidualNorm)) {
        throw std::runtime_error("BiCGStab initial residual is not finite");
    }
    replaceRecurrenceResidual();

    checkCuda(cudaMemcpyAsync(
        context->d_r0_hat, context->d_r, N * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToDevice, context->stream
    ), "initializing shadow residual");
    double shadowResidualNorm = res_norm;
    if (initiallyConverged) {
        const double finalResidual = std::max(
            verifiedTrueResidual, verifiedScaledResidual
        );
        std::cout << "  -> BiCGStab converged immediately (Warm Start). Final res: "
                  << finalResidual << " (true=" << verifiedTrueResidual
                  << ", Jacobi-scaled=" << verifiedScaledResidual << ")" << std::endl;
        if (stats) {
            stats->iterations = 0;
            stats->finalResidual = finalResidual;
            stats->finalTrueResidual = verifiedTrueResidual;
            stats->finalJacobiScaledResidual = verifiedScaledResidual;
            stats->converged = true;
        }
        return true;
    }

    int iter = 0;
    bool converged = false;
    bool finalResidualIsVerified = false;
    int breakdownRestarts = 0;
    constexpr int maxBreakdownRestarts = 8;

    auto resetKrylovState = [&]() {
        checkCuda(cudaMemcpyAsync(
            context->d_r0_hat, context->d_r, N * sizeof(cuDoubleComplex),
            cudaMemcpyDeviceToDevice, context->stream
        ), "restarting shadow residual");
        checkCuda(cudaMemsetAsync(
            context->d_p, 0, N * sizeof(cuDoubleComplex), context->stream
        ), "restarting search direction");
        checkCuda(cudaMemsetAsync(
            context->d_v, 0, N * sizeof(cuDoubleComplex), context->stream
        ), "restarting Krylov vector");
        shadowResidualNorm = res_norm;
        rho_prev = make_cuDoubleComplex(1.0, 0.0);
        alpha = make_cuDoubleComplex(1.0, 0.0);
        omega = make_cuDoubleComplex(1.0, 0.0);
    };

    auto recoverFromBreakdown = [&]() {
        const bool residualConverged = verifyTrueResidual();
        if (!std::isfinite(verifiedTrueResidual)
            || !std::isfinite(verifiedScaledResidual)
            || !std::isfinite(verifiedScaledResidualNorm)) {
            return false;
        }
        if (residualConverged) {
            converged = true;
            finalResidualIsVerified = true;
            return false;
        }
        if (breakdownRestarts >= maxBreakdownRestarts) {
            return false;
        }
        ++breakdownRestarts;
        replaceRecurrenceResidual();
        resetKrylovState();
        return true;
    };

    for (iter = 1; iter <= maxIter; ++iter) {
        cuDoubleComplex rho;
        checkCublas(cublasZdotc(
            context->cublasH, N, context->d_r0_hat, 1, context->d_r, 1, &rho
        ), "cublasZdotc rho");

        if (scaledBreakdown(rho, shadowResidualNorm, res_norm)
            || !finiteComplex(omega) || cuCabs(omega) == 0.0) {
            if (recoverFromBreakdown()) {
                continue;
            }
            break;
        }

        cuDoubleComplex beta = cuCmul(cuCdiv(rho, rho_prev), cuCdiv(alpha, omega));
        if (!finiteComplex(beta)) {
            if (recoverFromBreakdown()) {
                continue;
            }
            break;
        }

        updateSearchDirectionKernel<<<blocks, threads, 0, context->stream>>>(
            N, context->d_r, beta, omega, context->d_v, context->d_p
        );
        checkCuda(cudaGetLastError(), "updating BiCGStab search direction");

        SpMV_LeftJacobi(context->d_p, context->d_v);

        cuDoubleComplex r0_v;
        checkCublas(cublasZdotc(
            context->cublasH, N, context->d_r0_hat, 1, context->d_v, 1, &r0_v
        ), "cublasZdotc r0v");
        if (!finiteComplex(r0_v) || cuCabs(r0_v) == 0.0) {
            if (recoverFromBreakdown()) {
                continue;
            }
            break;
        }
        const cuDoubleComplex candidateAlpha = cuCdiv(rho, r0_v);
        if (!finiteComplex(candidateAlpha)) {
            if (recoverFromBreakdown()) {
                continue;
            }
            break;
        }

        const long double candidateMagnitude = std::hypot(
            static_cast<long double>(cuCreal(candidateAlpha)),
            static_cast<long double>(cuCimag(candidateAlpha))
        );
        const long double previousMagnitude = std::max(
            1.0L,
            std::hypot(
                static_cast<long double>(cuCreal(alpha)),
                static_cast<long double>(cuCimag(alpha))
            )
        );
        const long double suspiciousGrowth = 1.0L / std::sqrt(
            static_cast<long double>(std::numeric_limits<double>::epsilon())
        );
        if (candidateMagnitude > suspiciousGrowth * previousMagnitude) {
            double vNorm = 0.0;
            checkCublas(cublasDznrm2(
                context->cublasH, N, context->d_v, 1, &vNorm
            ), "cublasDznrm2 suspicious Krylov vector");
            if (scaledBreakdown(r0_v, shadowResidualNorm, vNorm)) {
                if (recoverFromBreakdown()) {
                    continue;
                }
                break;
            }
        }
        alpha = candidateAlpha;

        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-cuCreal(alpha), -cuCimag(alpha));
        checkCublas(cublasZaxpy(
            context->cublasH, N, &neg_alpha, context->d_v, 1, context->d_r, 1
        ), "cublasZaxpy alpha residual");

        checkCublas(cublasDznrm2(
            context->cublasH, N, context->d_r, 1, &res_norm
        ), "cublasDznrm2 intermediate residual");
        if (relativeResidual(res_norm, jacobiScaledNormB) < tol) {
            checkCublas(cublasZaxpy(
                context->cublasH, N, &alpha, context->d_p, 1, d_x, 1
            ), "cublasZaxpy alpha solution");
            const bool residualConverged = verifyTrueResidual();
            if (residualConverged) {
                converged = true;
                finalResidualIsVerified = true;
                break;
            }
            if (!std::isfinite(verifiedTrueResidual)
                || !std::isfinite(verifiedScaledResidual)
                || !std::isfinite(verifiedScaledResidualNorm)) {
                break;
            }
            replaceRecurrenceResidual();
            resetKrylovState();
            continue;
        }

        SpMV_LeftJacobi(context->d_r, context->d_t);

        cuDoubleComplex ts, tt;
        checkCublas(cublasZdotc(
            context->cublasH, N, context->d_t, 1, context->d_r, 1, &ts
        ), "cublasZdotc ts");
        checkCublas(cublasZdotc(
            context->cublasH, N, context->d_t, 1, context->d_t, 1, &tt
        ), "cublasZdotc tt");
        if (!finiteComplex(ts) || !finiteComplex(tt) || cuCreal(tt) <= 0.0) {
            if (recoverFromBreakdown()) {
                continue;
            }
            break;
        }
        const double tNorm = std::sqrt(cuCreal(tt));
        if (scaledBreakdown(ts, tNorm, res_norm)) {
            if (recoverFromBreakdown()) {
                continue;
            }
            break;
        }
        omega = cuCdiv(ts, tt);
        if (!finiteComplex(omega) || cuCabs(omega) == 0.0) {
            if (recoverFromBreakdown()) {
                continue;
            }
            break;
        }

        updateSolutionAndResidualKernel<<<blocks, threads, 0, context->stream>>>(
            N, alpha, omega, context->d_p, context->d_r,
            context->d_t, d_x, context->d_r
        );
        checkCuda(cudaGetLastError(), "updating BiCGStab solution and residual");

        rho_prev = rho;

        checkCublas(cublasDznrm2(
            context->cublasH, N, context->d_r, 1, &res_norm
        ), "cublasDznrm2 residual");
        if (!std::isfinite(res_norm)) {
            break;
        }
        const bool recurrenceResidualCandidate =
            relativeResidual(res_norm, jacobiScaledNormB) < tol;
        if (recurrenceResidualCandidate || iter % 200 == 0) {
            const bool residualConverged = verifyTrueResidual();
            if (residualConverged) {
                converged = true;
                finalResidualIsVerified = true;
                break;
            }
            if (!std::isfinite(verifiedTrueResidual)
                || !std::isfinite(verifiedScaledResidual)
                || !std::isfinite(verifiedScaledResidualNorm)) {
                break;
            }
            if (recurrenceResidualCandidate) {
                replaceRecurrenceResidual();
                resetKrylovState();
            }
        }

        if (iter % 500 == 0) {
            std::cout << "  [Left-Jacobi BiCGStab] Iter " << iter
                      << " - Scaled residual: "
                      << relativeResidual(res_norm, jacobiScaledNormB) << std::endl;
        }
    }

    if (!finalResidualIsVerified) {
        converged = verifyTrueResidual();
    }
    const double finalResidual = std::max(
        verifiedTrueResidual, verifiedScaledResidual
    );
    converged = converged
        && std::isfinite(finalResidual)
        && finalResidual < tol;
    int completedIterations = std::min(iter, maxIter);

    if (converged) {
        std::cout << "  -> BiCGStab converged in " << completedIterations
                  << " iterations. Final res: " << finalResidual
                  << " (true=" << verifiedTrueResidual
                  << ", Jacobi-scaled=" << verifiedScaledResidual << ")" << std::endl;
    } else {
        std::cerr << "  -> BiCGStab failed after " << completedIterations
                  << " iterations (maxIter=" << maxIter
                  << ", breakdown restarts=" << breakdownRestarts
                  << "). Final res: " << finalResidual
                  << " (true=" << verifiedTrueResidual
                  << ", Jacobi-scaled=" << verifiedScaledResidual << ")" << std::endl;
    }

    if (stats) {
        stats->iterations = completedIterations;
        stats->finalResidual = finalResidual;
        stats->finalTrueResidual = verifiedTrueResidual;
        stats->finalJacobiScaledResidual = verifiedScaledResidual;
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
    if (!context) {
        throw std::invalid_argument("GpuBiCGStabContext is required for CSR initialization");
    }
    if (h_csrRowPtr.empty()
        || h_csrRowPtr.size() - 1 > static_cast<size_t>(std::numeric_limits<int>::max())
        || h_csrColInd.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("BiCGStab CSR structure exceeds 32-bit capacity");
    }
    validate_context(
        context,
        static_cast<int>(h_csrRowPtr.size() - 1),
        static_cast<int>(h_csrColInd.size())
    );
    initialize_structure_if_needed(context, h_csrRowPtr, h_csrColInd);
}

void set_gpu_bicgstab_stream(GpuBiCGStabContext* context, cudaStream_t stream)
{
    if (!context) {
        throw std::runtime_error("GpuBiCGStabContext is required when setting a stream");
    }
    int currentDevice = -1;
    checkCuda(cudaGetDevice(&currentDevice), "querying current CUDA device");
    if (currentDevice != context->device) {
        throw std::runtime_error("BiCGStab context is bound to a different CUDA device");
    }
    context->stream = stream;
    checkCublas(cublasSetStream(context->cublasH, stream), "cublasSetStream");
    checkCusparse(cusparseSetStream(context->cusparseH, stream), "cusparseSetStream");
}

const int* get_gpu_bicgstab_row_ptr_device(const GpuBiCGStabContext* context)
{
    if (!context || !context->structureInitialized) {
        throw std::runtime_error("BiCGStab CSR structure is not initialized");
    }
    return context->d_csrRowPtr;
}

const int* get_gpu_bicgstab_col_ind_device(const GpuBiCGStabContext* context)
{
    if (!context || !context->structureInitialized) {
        throw std::runtime_error("BiCGStab CSR structure is not initialized");
    }
    return context->d_csrColInd;
}

cuDoubleComplex* get_gpu_bicgstab_matrix_values_device(GpuBiCGStabContext* context)
{
    if (!context || !context->structureInitialized) {
        throw std::runtime_error("BiCGStab CSR structure is not initialized");
    }
    return context->d_csrVal;
}

cuDoubleComplex* get_gpu_bicgstab_rhs_device(GpuBiCGStabContext* context)
{
    if (!context || !context->structureInitialized) {
        throw std::runtime_error("BiCGStab CSR structure is not initialized");
    }
    return context->d_b;
}

bool gpu_complex_solver_bicgstab_device(GpuBiCGStabContext* context,
                                        const cuDoubleComplex* d_b,
                                        cuDoubleComplex* d_x_io,
                                        bool rebuildPreconditioner,
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

    int currentDevice = -1;
    checkCuda(cudaGetDevice(&currentDevice), "querying current CUDA device");
    if (currentDevice != context->device) {
        throw std::runtime_error("BiCGStab context is bound to a different CUDA device");
    }

    if (!d_b || !d_x_io) {
        throw std::runtime_error("Device RHS and solution pointers must be non-null");
    }
    if (!(tol > 0.0) || !std::isfinite(tol) || maxIter <= 0) {
        throw std::runtime_error("BiCGStab tolerance and maxIter must be positive");
    }

    return solve_bicgstab_in_context(
        context, d_b, d_x_io, rebuildPreconditioner, tol, maxIter, stats
    );
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
    if (N <= 0 || h_b.size() != static_cast<size_t>(N) || x.size() != static_cast<size_t>(N)) {
        throw std::invalid_argument("BiCGStab RHS and solution sizes must match N");
    }
    if (!(tol > 0.0) || !std::isfinite(tol) || maxIter <= 0) {
        throw std::invalid_argument("BiCGStab tolerance and maxIter must be positive and finite");
    }
    if (h_csrRowPtr.size() != static_cast<size_t>(N + 1)) {
        throw std::invalid_argument("BiCGStab CSR row pointer size must be N+1");
    }
    if (h_csrVal.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("BiCGStab CSR nnz exceeds the 32-bit index range");
    }
    int nnz = static_cast<int>(h_csrVal.size());
    if (nnz <= 0 || h_csrColInd.size() != static_cast<size_t>(nnz)) {
        throw std::invalid_argument("BiCGStab CSR column and value sizes must match and be nonzero");
    }
    validate_context(context, N, nnz);
    std::unique_ptr<GpuBiCGStabContext, void(*)(GpuBiCGStabContext*)> ownedContext(
        nullptr, free_context
    );
    if (!context) {
        ownedContext.reset(allocate_context(N, nnz));
        context = ownedContext.get();
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

    checkCuda(cudaMemcpy(
        context->d_csrVal, c_csrVal.data(),
        nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice
    ), "copying CSR values");
    checkCuda(cudaMemcpy(
        context->d_b, c_b.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice
    ), "copying RHS");
    checkCuda(cudaMemcpy(
        context->d_x, c_x.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice
    ), "copying initial solution");
    bool converged = solve_bicgstab_in_context(
        context, context->d_b, context->d_x, true, tol, maxIter, stats
    );

    checkCuda(cudaMemcpy(
        c_x.data(), context->d_x, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost
    ), "copying solution to host");
    for (int i = 0; i < N; i++) {
        x[i] = std::complex<double>(cuCreal(c_x[i]), cuCimag(c_x[i]));
    }

    return converged;
}
