// include/femAssemble3d.h
#ifndef FEMASSEMBLE3D_H
#define FEMASSEMBLE3D_H

#include "mesh3d.h"
#include <vector>
#include <complex>

struct ComplexTriple {
    int row;
    int col;
    std::complex<double> val;
};

void femAssemble3D_Matrix(const HostMesh3D &mesh, double freq, const std::vector<double> &rho,
                          std::vector<std::complex<double>> &csrVal, 
                          std::vector<int> &csrColInd, 
                          std::vector<int> &csrRowPtr);

void femBuild3D_CSRStructure(const HostMesh3D &mesh,
                             std::vector<int> &csrColInd,
                             std::vector<int> &csrRowPtr);

void complexTripletToCSR3D(int N, const std::vector<ComplexTriple> &triplets,
                           std::vector<int> &csrRowPtr, 
                           std::vector<int> &csrColInd, 
                           std::vector<std::complex<double>> &csrVal);

#endif // FEMASSEMBLE3D_H
