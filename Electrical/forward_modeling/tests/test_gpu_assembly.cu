#include "../include/femAssemble3d.h"
#include "../include/mesh3d.h"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

extern "C" void gpu_assemble_component_matrices(
    int NE, int NX, int NY, int nnz,
    const double* d_hx, const double* d_hy, const double* d_hz,
    const double* d_rho, const int* d_meEdges,
    const int* d_csrRowPtr, const int* d_csrColInd,
    double* d_stiffness, double* d_conductivityMass,
    int* d_invalidFlag, cudaStream_t stream);
extern "C" void gpu_combine_component_matrices(
    int nnz, double freq, const double* d_stiffness,
    const double* d_conductivityMass, cuDoubleComplex* d_values,
    cudaStream_t stream);

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

    explicit DeviceBuffer(size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

} // namespace

int main()
{
    const std::vector<double> hx = {1.1, 2.3};
    const std::vector<double> hy = {0.7, 1.9};
    const std::vector<double> hz = {1.4, 3.1};
    const std::vector<double> rho = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};
    constexpr double frequency = 3.7;
    HostMesh3D mesh = generateMesh3D(2, 2, 2, hx, hy, hz);

    std::vector<int> colInd;
    std::vector<int> rowPtr;
    femBuild3D_CSRStructure(mesh, colInd, rowPtr);
    const int nnz = static_cast<int>(colInd.size());

    DeviceBuffer<double> dHx(hx.size()), dHy(hy.size()), dHz(hz.size()), dRho(rho.size());
    DeviceBuffer<int> dEdges(mesh.ME_Edges.size()), dRowPtr(rowPtr.size()), dColInd(colInd.size());
    DeviceBuffer<int> dInvalidFlag(1);
    DeviceBuffer<double> dStiffness(nnz), dMass(nnz);
    DeviceBuffer<cuDoubleComplex> dValues(nnz);

    check(cudaMemcpy(dHx.ptr, hx.data(), hx.size() * sizeof(double), cudaMemcpyHostToDevice), "copy hx");
    check(cudaMemcpy(dHy.ptr, hy.data(), hy.size() * sizeof(double), cudaMemcpyHostToDevice), "copy hy");
    check(cudaMemcpy(dHz.ptr, hz.data(), hz.size() * sizeof(double), cudaMemcpyHostToDevice), "copy hz");
    check(cudaMemcpy(dRho.ptr, rho.data(), rho.size() * sizeof(double), cudaMemcpyHostToDevice), "copy rho");
    check(cudaMemcpy(dEdges.ptr, mesh.ME_Edges.data(), mesh.ME_Edges.size() * sizeof(int), cudaMemcpyHostToDevice), "copy edges");
    check(cudaMemcpy(dRowPtr.ptr, rowPtr.data(), rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "copy row pointers");
    check(cudaMemcpy(dColInd.ptr, colInd.data(), colInd.size() * sizeof(int), cudaMemcpyHostToDevice), "copy columns");

    gpu_assemble_component_matrices(
        mesh.NE, mesh.NX, mesh.NY, nnz,
        dHx.ptr, dHy.ptr, dHz.ptr, dRho.ptr,
        dEdges.ptr, dRowPtr.ptr, dColInd.ptr,
        dStiffness.ptr, dMass.ptr, dInvalidFlag.ptr, nullptr
    );
    gpu_combine_component_matrices(
        nnz, frequency, dStiffness.ptr, dMass.ptr, dValues.ptr, nullptr
    );
    check(cudaDeviceSynchronize(), "GPU assembly");

    int invalidFlag = 0;
    check(cudaMemcpy(
        &invalidFlag, dInvalidFlag.ptr, sizeof(int), cudaMemcpyDeviceToHost
    ), "copy assembly validation flag");
    if (invalidFlag != 0) {
        std::cerr << "GPU assembly found a missing CSR entry\n";
        return 1;
    }

    std::vector<cuDoubleComplex> gpuValues(nnz);
    check(cudaMemcpy(
        gpuValues.data(), dValues.ptr, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost
    ), "copy matrix values");

    std::vector<std::complex<double>> referenceValues;
    std::vector<int> referenceCols;
    std::vector<int> referenceRows;
    femAssemble3D_Matrix(
        mesh, frequency, rho, referenceValues, referenceCols, referenceRows
    );

    double maxRelativeError = 0.0;
    for (int row = 0; row < mesh.NEdges; ++row) {
        for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; ++idx) {
            std::complex<double> expected(0.0, 0.0);
            auto begin = referenceCols.begin() + referenceRows[row];
            auto end = referenceCols.begin() + referenceRows[row + 1];
            auto found = std::lower_bound(begin, end, colInd[idx]);
            if (found != end && *found == colInd[idx]) {
                expected = referenceValues[found - referenceCols.begin()];
            }
            std::complex<double> actual(
                cuCreal(gpuValues[idx]), cuCimag(gpuValues[idx])
            );
            double scale = std::max(1.0, std::abs(expected));
            maxRelativeError = std::max(
                maxRelativeError, std::abs(actual - expected) / scale
            );
        }
    }

    if (maxRelativeError > 2e-12) {
        std::cerr << "GPU/CPU assembly mismatch: " << maxRelativeError << '\n';
        return 1;
    }
    return 0;
}
