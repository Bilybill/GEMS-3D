// include/boundaryConditions3d.h
#ifndef BOUNDARYCONDITIONS3D_H
#define BOUNDARYCONDITIONS3D_H

#include <cuComplex.h>
#include "mesh3d.h"
#include <vector>
#include <complex>

struct BoundaryConditionPlan;

BoundaryConditionPlan* createBoundaryConditionPlan3D(const HostMesh3D &mesh,
                                                     const std::vector<int> &csrColInd,
                                                     const std::vector<int> &csrRowPtr);

void destroyBoundaryConditionPlan3D(BoundaryConditionPlan* plan);

void getBoundaryConditionModifiedValueIndices3D(const BoundaryConditionPlan* plan,
                                                std::vector<int> &indices);

// 施加 3D 边界条件
// polarization = 1: Ex 极化; polarization = 2: Ey 极化
void applyBoundaryConditions3D(const HostMesh3D &mesh, double freq, double rho_bg,
                               std::vector<std::complex<double>> &csrVal, 
                               std::vector<int> &csrColInd, 
                               const std::vector<int> &csrRowPtr,
                               std::vector<std::complex<double>> &b,
                               int polarization,
                               const BoundaryConditionPlan* plan = nullptr);

void applyBoundaryConditions3DDevice(double freq, double rho_bg,
                                     cuDoubleComplex* d_csrVal,
                                     cuDoubleComplex* d_b,
                                     int polarization,
                                     const BoundaryConditionPlan* plan);

#endif // BOUNDARYCONDITIONS3D_H
