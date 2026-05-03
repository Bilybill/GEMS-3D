// src/boundaryConditions3d.cpp
#include "../include/boundaryConditions3d.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <algorithm>
#include <stdexcept>

extern "C" void gpu_apply_boundary_conditions(
    int boundaryEdgeCount,
    const int* d_boundaryEdgeFamilies,
    const int* d_boundaryEdgeZIndices,
    const double* d_zCoords,
    cuDoubleComplex k_num,
    int polarization,
    int couplingCount,
    const int* d_couplingRows,
    const int* d_couplingCsrIndices,
    const int* d_couplingBoundaryPos,
    int boundaryRowCount,
    const int* d_boundaryRowRows,
    const int* d_boundaryRowStarts,
    const int* d_boundaryRowEnds,
    const int* d_boundaryRowDiagIndices,
    const int* d_boundaryRowBoundaryPos,
    cuDoubleComplex* d_csrVal,
    cuDoubleComplex* d_b,
    cuDoubleComplex* d_bcValues);

namespace {

struct BoundaryCouplingEntry {
    int row;
    int csrIndex;
    int boundaryCol;
};

struct BoundaryRowEntry {
    int row;
    int rowStart;
    int rowEnd;
    int diagIndex;
};

struct BoundaryConditionPlanImpl {
    std::vector<double> zCoords;
    std::vector<unsigned char> isBoundary;
    std::vector<int> boundaryEdges;
    std::vector<int> boundaryLookup;
    std::vector<BoundaryCouplingEntry> boundaryCouplings;
    std::vector<BoundaryRowEntry> boundaryRows;
    std::vector<int> modifiedValueIndices;
    std::vector<int> boundaryEdgeFamilies;
    std::vector<int> boundaryEdgeZIndices;
    std::vector<int> boundaryRowBoundaryPos;

    int nEdges = 0;
    double* d_zCoords = nullptr;
    int* d_boundaryEdgeFamilies = nullptr;
    int* d_boundaryEdgeZIndices = nullptr;
    int* d_couplingRows = nullptr;
    int* d_couplingCsrIndices = nullptr;
    int* d_couplingBoundaryPos = nullptr;
    int* d_boundaryRowRows = nullptr;
    int* d_boundaryRowStarts = nullptr;
    int* d_boundaryRowEnds = nullptr;
    int* d_boundaryRowDiagIndices = nullptr;
    int* d_boundaryRowBoundaryPos = nullptr;
    cuDoubleComplex* d_bcValues = nullptr;
};

template <typename T>
void copyVectorToDevice(const std::vector<T>& host, T*& devicePtr)
{
    if (host.empty()) {
        devicePtr = nullptr;
        return;
    }
    cudaMalloc(reinterpret_cast<void**>(&devicePtr), host.size() * sizeof(T));
    cudaMemcpy(devicePtr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
}

void freePlanDeviceBuffers(BoundaryConditionPlanImpl& plan)
{
    cudaFree(plan.d_zCoords);
    cudaFree(plan.d_boundaryEdgeFamilies);
    cudaFree(plan.d_boundaryEdgeZIndices);
    cudaFree(plan.d_couplingRows);
    cudaFree(plan.d_couplingCsrIndices);
    cudaFree(plan.d_couplingBoundaryPos);
    cudaFree(plan.d_boundaryRowRows);
    cudaFree(plan.d_boundaryRowStarts);
    cudaFree(plan.d_boundaryRowEnds);
    cudaFree(plan.d_boundaryRowDiagIndices);
    cudaFree(plan.d_boundaryRowBoundaryPos);
    cudaFree(plan.d_bcValues);

    plan.d_zCoords = nullptr;
    plan.d_boundaryEdgeFamilies = nullptr;
    plan.d_boundaryEdgeZIndices = nullptr;
    plan.d_couplingRows = nullptr;
    plan.d_couplingCsrIndices = nullptr;
    plan.d_couplingBoundaryPos = nullptr;
    plan.d_boundaryRowRows = nullptr;
    plan.d_boundaryRowStarts = nullptr;
    plan.d_boundaryRowEnds = nullptr;
    plan.d_boundaryRowDiagIndices = nullptr;
    plan.d_boundaryRowBoundaryPos = nullptr;
    plan.d_bcValues = nullptr;
}

void initializePlanDeviceBuffers(BoundaryConditionPlanImpl& plan)
{
    copyVectorToDevice(plan.zCoords, plan.d_zCoords);
    copyVectorToDevice(plan.boundaryEdgeFamilies, plan.d_boundaryEdgeFamilies);
    copyVectorToDevice(plan.boundaryEdgeZIndices, plan.d_boundaryEdgeZIndices);

    std::vector<int> couplingRows;
    std::vector<int> couplingCsrIndices;
    std::vector<int> couplingBoundaryPos;
    couplingRows.reserve(plan.boundaryCouplings.size());
    couplingCsrIndices.reserve(plan.boundaryCouplings.size());
    couplingBoundaryPos.reserve(plan.boundaryCouplings.size());
    for (const auto& coupling : plan.boundaryCouplings) {
        couplingRows.push_back(coupling.row);
        couplingCsrIndices.push_back(coupling.csrIndex);
        couplingBoundaryPos.push_back(plan.boundaryLookup[coupling.boundaryCol]);
    }
    copyVectorToDevice(couplingRows, plan.d_couplingRows);
    copyVectorToDevice(couplingCsrIndices, plan.d_couplingCsrIndices);
    copyVectorToDevice(couplingBoundaryPos, plan.d_couplingBoundaryPos);

    std::vector<int> boundaryRowRows;
    std::vector<int> boundaryRowStarts;
    std::vector<int> boundaryRowEnds;
    std::vector<int> boundaryRowDiagIndices;
    boundaryRowRows.reserve(plan.boundaryRows.size());
    boundaryRowStarts.reserve(plan.boundaryRows.size());
    boundaryRowEnds.reserve(plan.boundaryRows.size());
    boundaryRowDiagIndices.reserve(plan.boundaryRows.size());
    for (const auto& rowEntry : plan.boundaryRows) {
        boundaryRowRows.push_back(rowEntry.row);
        boundaryRowStarts.push_back(rowEntry.rowStart);
        boundaryRowEnds.push_back(rowEntry.rowEnd);
        boundaryRowDiagIndices.push_back(rowEntry.diagIndex);
    }
    copyVectorToDevice(boundaryRowRows, plan.d_boundaryRowRows);
    copyVectorToDevice(boundaryRowStarts, plan.d_boundaryRowStarts);
    copyVectorToDevice(boundaryRowEnds, plan.d_boundaryRowEnds);
    copyVectorToDevice(boundaryRowDiagIndices, plan.d_boundaryRowDiagIndices);
    copyVectorToDevice(plan.boundaryRowBoundaryPos, plan.d_boundaryRowBoundaryPos);

    if (!plan.boundaryEdges.empty()) {
        cudaMalloc(reinterpret_cast<void**>(&plan.d_bcValues), plan.boundaryEdges.size() * sizeof(cuDoubleComplex));
    }
}

BoundaryConditionPlanImpl buildBoundaryConditionPlanImpl(const HostMesh3D &mesh,
                                                         const std::vector<int> &csrColInd,
                                                         const std::vector<int> &csrRowPtr)
{
    BoundaryConditionPlanImpl plan;

    int NX = mesh.NX;
    int NY = mesh.NY;
    int NZ = mesh.NZ;

    int nEx = NX * (NY + 1) * (NZ + 1);
    int nEy = (NX + 1) * NY * (NZ + 1);

    int offset_X = 0;
    int offset_Y = nEx;
    int offset_Z = nEx + nEy;

    plan.nEdges = mesh.NEdges;
    plan.zCoords.assign(NZ + 1, 0.0);
    for (int k = 0; k < NZ; ++k) {
        plan.zCoords[k + 1] = plan.zCoords[k] + mesh.hz[k];
    }

    plan.isBoundary.assign(mesh.NEdges, 0);
    plan.boundaryLookup.assign(mesh.NEdges, -1);

    auto markBoundary = [&](int edgeId) {
        if (!plan.isBoundary[edgeId]) {
            plan.isBoundary[edgeId] = 1;
            plan.boundaryLookup[edgeId] = static_cast<int>(plan.boundaryEdges.size());
            plan.boundaryEdges.push_back(edgeId);
        }
    };

    for (int k_idx = 0; k_idx <= NZ; ++k_idx) {
        for (int j = 0; j <= NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if (k_idx == 0 || k_idx == NZ || j == 0 || j == NY) {
                    markBoundary(offset_X + i + j * NX + k_idx * NX * (NY + 1));
                }
            }
        }
    }

    for (int k_idx = 0; k_idx <= NZ; ++k_idx) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i <= NX; ++i) {
                if (k_idx == 0 || k_idx == NZ || i == 0 || i == NX) {
                    markBoundary(offset_Y + i + j * (NX + 1) + k_idx * (NX + 1) * NY);
                }
            }
        }
    }

    for (int k_idx = 0; k_idx < NZ; ++k_idx) {
        for (int j = 0; j <= NY; ++j) {
            for (int i = 0; i <= NX; ++i) {
                if (i == 0 || i == NX || j == 0 || j == NY) {
                    markBoundary(offset_Z + i + j * (NX + 1) + k_idx * (NX + 1) * (NY + 1));
                }
            }
        }
    }

    for (int row = 0; row < mesh.NEdges; ++row) {
        if (plan.isBoundary[row]) {
            int diagIndex = -1;
            for (int idx = csrRowPtr[row]; idx < csrRowPtr[row + 1]; ++idx) {
                if (csrColInd[idx] == row) {
                    diagIndex = idx;
                    break;
                }
            }
            plan.boundaryRows.push_back({row, csrRowPtr[row], csrRowPtr[row + 1], diagIndex});
            plan.boundaryRowBoundaryPos.push_back(plan.boundaryLookup[row]);
            continue;
        }

        for (int idx = csrRowPtr[row]; idx < csrRowPtr[row + 1]; ++idx) {
            int col = csrColInd[idx];
            if (plan.isBoundary[col]) {
                plan.boundaryCouplings.push_back({row, idx, col});
            }
        }
    }

    plan.modifiedValueIndices.reserve(plan.boundaryCouplings.size() + plan.boundaryRows.size() * 8);
    for (const auto& coupling : plan.boundaryCouplings) {
        plan.modifiedValueIndices.push_back(coupling.csrIndex);
    }
    for (const auto& rowEntry : plan.boundaryRows) {
        for (int idx = rowEntry.rowStart; idx < rowEntry.rowEnd; ++idx) {
            plan.modifiedValueIndices.push_back(idx);
        }
    }
    std::sort(plan.modifiedValueIndices.begin(), plan.modifiedValueIndices.end());
    plan.modifiedValueIndices.erase(
        std::unique(plan.modifiedValueIndices.begin(), plan.modifiedValueIndices.end()),
        plan.modifiedValueIndices.end()
    );

    plan.boundaryEdgeFamilies.reserve(plan.boundaryEdges.size());
    plan.boundaryEdgeZIndices.reserve(plan.boundaryEdges.size());
    for (int edgeId : plan.boundaryEdges) {
        if (edgeId < offset_Y) {
            int local = edgeId - offset_X;
            int k_idx = local / (NX * (NY + 1));
            plan.boundaryEdgeFamilies.push_back(0);
            plan.boundaryEdgeZIndices.push_back(k_idx);
        } else if (edgeId < offset_Z) {
            int local = edgeId - offset_Y;
            int k_idx = local / ((NX + 1) * NY);
            plan.boundaryEdgeFamilies.push_back(1);
            plan.boundaryEdgeZIndices.push_back(k_idx);
        } else {
            plan.boundaryEdgeFamilies.push_back(2);
            plan.boundaryEdgeZIndices.push_back(0);
        }
    }

    initializePlanDeviceBuffers(plan);

    return plan;
}

const BoundaryConditionPlanImpl* unwrapPlan(const BoundaryConditionPlan* plan)
{
    return reinterpret_cast<const BoundaryConditionPlanImpl*>(plan);
}

} // namespace

struct BoundaryConditionPlan {};

BoundaryConditionPlan* createBoundaryConditionPlan3D(const HostMesh3D &mesh,
                                                     const std::vector<int> &csrColInd,
                                                     const std::vector<int> &csrRowPtr)
{
    auto impl = buildBoundaryConditionPlanImpl(mesh, csrColInd, csrRowPtr);
    return reinterpret_cast<BoundaryConditionPlan*>(new BoundaryConditionPlanImpl(std::move(impl)));
}

void destroyBoundaryConditionPlan3D(BoundaryConditionPlan* plan)
{
    auto* impl = reinterpret_cast<BoundaryConditionPlanImpl*>(plan);
    freePlanDeviceBuffers(*impl);
    delete impl;
}

void getBoundaryConditionModifiedValueIndices3D(const BoundaryConditionPlan* plan,
                                                std::vector<int> &indices)
{
    indices = unwrapPlan(plan)->modifiedValueIndices;
}

void applyBoundaryConditions3D(const HostMesh3D &mesh, double freq, double rho_bg,
                               std::vector<std::complex<double>> &csrVal,
                               std::vector<int> &csrColInd,
                               const std::vector<int> &csrRowPtr,
                               std::vector<std::complex<double>> &b,
                               int polarization,
                               const BoundaryConditionPlan* planHandle)
{
    std::unique_ptr<BoundaryConditionPlan, void(*)(BoundaryConditionPlan*)> ownedPlan(nullptr, destroyBoundaryConditionPlan3D);
    if (!planHandle) {
        ownedPlan.reset(createBoundaryConditionPlan3D(mesh, csrColInd, csrRowPtr));
        planHandle = ownedPlan.get();
    }
    const auto* plan = unwrapPlan(planHandle);

    double mu = 4e-7 * M_PI;
    double w = 2.0 * M_PI * freq;
    double sigma = 1.0 / rho_bg;
    std::complex<double> neg_i(0.0, -1.0);
    std::complex<double> k_num = std::sqrt(neg_i * w * mu * sigma);

    if (static_cast<int>(b.size()) != mesh.NEdges) {
        b.resize(mesh.NEdges);
    }
    std::fill(b.begin(), b.end(), std::complex<double>(0.0, 0.0));

    std::vector<std::complex<double>> bc_values(plan->boundaryEdges.size(), {0.0, 0.0});

    int NX = mesh.NX;
    int NY = mesh.NY;
    int NZ = mesh.NZ;
    int nEx = NX * (NY + 1) * (NZ + 1);
    int nEy = (NX + 1) * NY * (NZ + 1);
    int offset_Y = nEx;
    int offset_Z = nEx + nEy;

    for (int edgeId : plan->boundaryEdges) {
        int boundaryPos = plan->boundaryLookup[edgeId];
        if (edgeId < offset_Y) {
            if (polarization == 1) {
                int local = edgeId;
                int k_idx = local / (NX * (NY + 1));
                bc_values[boundaryPos] = std::exp(neg_i * k_num * plan->zCoords[k_idx]);
            }
        } else if (edgeId < offset_Z) {
            if (polarization == 2) {
                int local = edgeId - offset_Y;
                int k_idx = local / ((NX + 1) * NY);
                bc_values[boundaryPos] = std::exp(neg_i * k_num * plan->zCoords[k_idx]);
            }
        }
    }

    for (const auto& entry : plan->boundaryCouplings) {
        b[entry.row] -= csrVal[entry.csrIndex] * bc_values[plan->boundaryLookup[entry.boundaryCol]];
        csrVal[entry.csrIndex] = {0.0, 0.0};
    }

    for (const auto& rowEntry : plan->boundaryRows) {
        b[rowEntry.row] = bc_values[plan->boundaryLookup[rowEntry.row]];
        for (int idx = rowEntry.rowStart; idx < rowEntry.rowEnd; ++idx) {
            csrVal[idx] = {0.0, 0.0};
        }
        if (rowEntry.diagIndex >= 0) {
            csrVal[rowEntry.diagIndex] = {1.0, 0.0};
        }
    }

    std::cout << "Applied 3D Boundary Conditions for Polarization " << polarization << " (Symmetric Elimination)" << std::endl;
}

void applyBoundaryConditions3DDevice(double freq, double rho_bg,
                                     cuDoubleComplex* d_csrVal,
                                     cuDoubleComplex* d_b,
                                     int polarization,
                                     const BoundaryConditionPlan* planHandle)
{
    const auto* plan = unwrapPlan(planHandle);
    if (!plan) {
        throw std::runtime_error("BoundaryConditionPlan is required for device-side BC application");
    }

    double mu = 4e-7 * M_PI;
    double w = 2.0 * M_PI * freq;
    double sigma = 1.0 / rho_bg;
    std::complex<double> neg_i(0.0, -1.0);
    std::complex<double> k_num_host = std::sqrt(neg_i * w * mu * sigma);
    cuDoubleComplex k_num = make_cuDoubleComplex(k_num_host.real(), k_num_host.imag());

    cudaMemset(d_b, 0, plan->nEdges * sizeof(cuDoubleComplex));
    if (!plan->boundaryEdges.empty()) {
        cudaMemset(plan->d_bcValues, 0, plan->boundaryEdges.size() * sizeof(cuDoubleComplex));
    }

    gpu_apply_boundary_conditions(
        static_cast<int>(plan->boundaryEdges.size()),
        plan->d_boundaryEdgeFamilies,
        plan->d_boundaryEdgeZIndices,
        plan->d_zCoords,
        k_num,
        polarization,
        static_cast<int>(plan->boundaryCouplings.size()),
        plan->d_couplingRows,
        plan->d_couplingCsrIndices,
        plan->d_couplingBoundaryPos,
        static_cast<int>(plan->boundaryRows.size()),
        plan->d_boundaryRowRows,
        plan->d_boundaryRowStarts,
        plan->d_boundaryRowEnds,
        plan->d_boundaryRowDiagIndices,
        plan->d_boundaryRowBoundaryPos,
        d_csrVal,
        d_b,
        plan->d_bcValues
    );
}
