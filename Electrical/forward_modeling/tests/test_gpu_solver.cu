#include "../include/gpuIterativeSolver.h"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;

    explicit DeviceBuffer(size_t count)
    {
        check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

std::vector<std::complex<double>> multiply(
    const std::vector<int>& rowPtr,
    const std::vector<int>& colInd,
    const std::vector<std::complex<double>>& values,
    const std::vector<std::complex<double>>& x)
{
    std::vector<std::complex<double>> result(x.size(), {0.0, 0.0});
    for (size_t row = 0; row < x.size(); ++row) {
        for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; ++idx) {
            result[row] += values[idx] * x[colInd[idx]];
        }
    }
    return result;
}

std::vector<cuDoubleComplex> toCudaComplex(
    const std::vector<std::complex<double>>& values)
{
    std::vector<cuDoubleComplex> result(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        result[index] = make_cuDoubleComplex(values[index].real(), values[index].imag());
    }
    return result;
}

double maxError(
    const std::vector<cuDoubleComplex>& actual,
    const std::vector<std::complex<double>>& expected)
{
    double error = 0.0;
    for (size_t index = 0; index < expected.size(); ++index) {
        std::complex<double> value(cuCreal(actual[index]), cuCimag(actual[index]));
        error = std::max(error, std::abs(value - expected[index]));
    }
    return error;
}

} // namespace

int main()
{
    constexpr int n = 4;
    const std::vector<int> rowPtr = {0, 2, 5, 8, 10};
    const std::vector<int> colInd = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    const std::vector<std::complex<double>> values = {
        {4.0, 1.0}, {1.0, -0.2},
        {2.0, 0.1}, {5.0, -0.5}, {1.0, 0.3},
        {-1.0, 0.2}, {3.0, 2.0}, {1.0, -0.4},
        {1.0, 0.5}, {2.0, -1.0},
    };
    const std::vector<std::complex<double>> expected1 = {
        {1.0, 0.5}, {-2.0, 1.0}, {0.25, -0.75}, {3.0, 0.2},
    };
    const std::vector<std::complex<double>> expected2 = {
        {-0.5, 1.5}, {1.25, -0.5}, {-2.0, 0.1}, {0.75, -1.0},
    };

    auto deviceValues = toCudaComplex(values);
    auto rhs1 = toCudaComplex(multiply(rowPtr, colInd, values, expected1));
    auto rhs2 = toCudaComplex(multiply(rowPtr, colInd, values, expected2));
    constexpr double matrixScale = 1e-120;
    std::vector<std::complex<double>> scaledValues(values.size());
    std::transform(
        values.begin(), values.end(), scaledValues.begin(),
        [matrixScale](const std::complex<double>& value) { return matrixScale * value; }
    );
    auto deviceScaledValues = toCudaComplex(scaledValues);
    auto scaledRhs = toCudaComplex(multiply(rowPtr, colInd, scaledValues, expected1));
    std::vector<cuDoubleComplex> actual1(n), actual2(n), actualScaled(n);

    DeviceBuffer<cuDoubleComplex> dRhs1(n), dRhs2(n), dScaledRhs(n);
    DeviceBuffer<cuDoubleComplex> dX1(n), dX2(n), dScaledX(n);
    auto* context = create_gpu_bicgstab_context(n, static_cast<int>(values.size()));
    initialize_gpu_bicgstab_structure(context, rowPtr, colInd);

    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    set_gpu_bicgstab_stream(context, stream);

    check(cudaMemcpyAsync(
        get_gpu_bicgstab_matrix_values_device(context), deviceValues.data(),
        deviceValues.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream
    ), "copy matrix values");
    check(cudaMemcpyAsync(
        dRhs1.ptr, rhs1.data(), rhs1.size() * sizeof(cuDoubleComplex),
        cudaMemcpyHostToDevice, stream
    ), "copy first RHS");
    check(cudaMemcpyAsync(
        dRhs2.ptr, rhs2.data(), rhs2.size() * sizeof(cuDoubleComplex),
        cudaMemcpyHostToDevice, stream
    ), "copy second RHS");
    check(cudaMemsetAsync(dX1.ptr, 0, n * sizeof(cuDoubleComplex), stream), "clear first solution");
    check(cudaMemsetAsync(dX2.ptr, 0, n * sizeof(cuDoubleComplex), stream), "clear second solution");

    SolverStats stats1;
    SolverStats stats2;
    bool converged1 = gpu_complex_solver_bicgstab_device(
        context, dRhs1.ptr, dX1.ptr, true, 1e-11, 1000, &stats1
    );
    bool converged2 = gpu_complex_solver_bicgstab_device(
        context, dRhs2.ptr, dX2.ptr, false, 1e-11, 1000, &stats2
    );
    check(cudaMemcpyAsync(
        actual1.data(), dX1.ptr, n * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToHost, stream
    ), "copy first solution");
    check(cudaMemcpyAsync(
        actual2.data(), dX2.ptr, n * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToHost, stream
    ), "copy second solution");
    check(cudaStreamSynchronize(stream), "synchronize solver test");

    check(cudaMemcpyAsync(
        get_gpu_bicgstab_matrix_values_device(context), deviceScaledValues.data(),
        deviceScaledValues.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream
    ), "copy scaled matrix values");
    check(cudaMemcpyAsync(
        dScaledRhs.ptr, scaledRhs.data(), scaledRhs.size() * sizeof(cuDoubleComplex),
        cudaMemcpyHostToDevice, stream
    ), "copy scaled RHS");
    check(cudaMemsetAsync(
        dScaledX.ptr, 0, n * sizeof(cuDoubleComplex), stream
    ), "clear scaled solution");
    SolverStats scaledStats;
    bool scaledConverged = gpu_complex_solver_bicgstab_device(
        context, dScaledRhs.ptr, dScaledX.ptr, true, 1e-11, 1000, &scaledStats
    );
    check(cudaMemcpyAsync(
        actualScaled.data(), dScaledX.ptr, n * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToHost, stream
    ), "copy scaled solution");
    check(cudaStreamSynchronize(stream), "synchronize scaled solver test");

    auto nonFiniteRhs = scaledRhs;
    nonFiniteRhs.front() = make_cuDoubleComplex(
        std::numeric_limits<double>::quiet_NaN(), 0.0
    );
    check(cudaMemcpyAsync(
        dScaledRhs.ptr, nonFiniteRhs.data(), nonFiniteRhs.size() * sizeof(cuDoubleComplex),
        cudaMemcpyHostToDevice, stream
    ), "copy non-finite RHS");
    bool rejectedNonFiniteRhs = false;
    try {
        gpu_complex_solver_bicgstab_device(
            context, dScaledRhs.ptr, dScaledX.ptr, false, 1e-11, 1000, nullptr
        );
    } catch (const std::runtime_error& error) {
        rejectedNonFiniteRhs = std::string(error.what()).find("RHS norm is not finite")
            != std::string::npos;
    }

    double error1 = maxError(actual1, expected1);
    double error2 = maxError(actual2, expected2);
    double scaledError = maxError(actualScaled, expected1);
    destroy_gpu_bicgstab_context(context);
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");

    if (!converged1 || !converged2 || !scaledConverged || !rejectedNonFiniteRhs
        || error1 > 1e-8 || error2 > 1e-8 || scaledError > 1e-8) {
        std::cerr << "BiCGStab regression failed: converged=" << converged1 << ','
                  << converged2 << ',' << scaledConverged
                  << " rejected_nonfinite=" << rejectedNonFiniteRhs
                  << " errors=" << error1 << ',' << error2 << ',' << scaledError
                  << " residuals=" << stats1.finalResidual << ','
                  << stats2.finalResidual << ',' << scaledStats.finalResidual << '\n';
        return 1;
    }
    return 0;
}
