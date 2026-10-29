// src/femAssemble3d.cpp
#include "../include/femAssemble3d.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

void complexTripletToCSR3D(int N, const std::vector<ComplexTriple> &triplets,
                           std::vector<int> &csrRowPtr, 
                           std::vector<int> &csrColInd, 
                           std::vector<std::complex<double>> &csrVal)
{
    std::vector<ComplexTriple> sorted = triplets;
    std::sort(sorted.begin(), sorted.end(), [](const ComplexTriple &a, const ComplexTriple &b){
        if (a.row == b.row) return a.col < b.col;
        return a.row < b.row;
    });

    std::vector<ComplexTriple> uniqueNZ;
    uniqueNZ.reserve(sorted.size());
    for (size_t i = 0; i < sorted.size(); i++) {
        if (i == 0 || sorted[i].row != sorted[i-1].row || sorted[i].col != sorted[i-1].col) {
            uniqueNZ.push_back(sorted[i]);
        } else {
            uniqueNZ.back().val += sorted[i].val;
        }
    }

    if (N < 0 || uniqueNZ.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CSR dimensions exceed the 32-bit index range");
    }
    int nz = static_cast<int>(uniqueNZ.size());
    csrVal.resize(nz);
    csrColInd.resize(nz);
    csrRowPtr.assign(N + 1, 0);

    for (const auto &nzItem : uniqueNZ) {
        csrRowPtr[nzItem.row + 1]++;
    }
    for (int i = 1; i <= N; i++) {
        csrRowPtr[i] += csrRowPtr[i - 1];
    }

    std::vector<int> countPerRow(N, 0);
    for (const auto &nzItem : uniqueNZ) {
        int r = nzItem.row;
        int dest = csrRowPtr[r] + countPerRow[r];
        csrColInd[dest] = nzItem.col;
        csrVal[dest] = nzItem.val;
        countPerRow[r]++;
    }
}

void computeLocalMatrix3D(double dx, double dy, double dz, double freq, double rho_val, std::complex<double> Ke[12][12]) {
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)
        || dx <= 0.0 || dy <= 0.0 || dz <= 0.0
        || !std::isfinite(freq) || freq <= 0.0
        || !std::isfinite(rho_val) || rho_val <= 0.0) {
        throw std::invalid_argument(
            "local matrix dimensions, frequency, and resistivity must be positive and finite"
        );
    }
    double mu = 4e-7 * M_PI;
    double w = 2.0 * M_PI * freq;
    std::complex<double> m(0.0, w * mu);     
    std::complex<double> oneOverM = 1.0 / m; 
    double sigma = 1.0 / rho_val;            

    for(int i=0; i<12; i++)
        for(int j=0; j<12; j++)
            Ke[i][j] = std::complex<double>(0.0, 0.0);

    double gp[2] = {0.5 - 0.5/sqrt(3.0), 0.5 + 0.5/sqrt(3.0)}; 
    double gw = 0.5; 

    for(int ig=0; ig<2; ig++) {
        for(int jg=0; jg<2; jg++) {
            for(int kg=0; kg<2; kg++) {
                double x = gp[ig] * dx, y = gp[jg] * dy, z = gp[kg] * dz;
                double dV = (dx * dy * dz) * gw * gw * gw;
                double xi = x/dx, yi = y/dy, zi = z/dz;
                
                double Nx[12]={0}, Ny[12]={0}, Nz[12]={0};
                Nx[0] = (1-yi)*(1-zi); Nx[2] = yi*(1-zi); Nx[4] = (1-yi)*zi; Nx[6] = yi*zi;
                Ny[1] = xi*(1-zi); Ny[3] = (1-xi)*(1-zi); Ny[5] = xi*zi; Ny[7] = (1-xi)*zi;
                Nz[8] = (1-xi)*(1-yi); Nz[9] = xi*(1-yi); Nz[10] = xi*yi; Nz[11] = (1-xi)*yi;

                double Cx[12]={0}, Cy[12]={0}, Cz[12]={0};
                
                // 【修复核心 1】修正了 X 族边 Cz 的正负号
                Cy[0] = -(1-yi)/dz; Cz[0] =  (1-zi)/dy;
                Cy[2] = -yi/dz;     Cz[2] = -(1-zi)/dy;
                Cy[4] =  (1-yi)/dz; Cz[4] =  zi/dy;
                Cy[6] =  yi/dz;     Cz[6] = -zi/dy;

                // Y 族边的旋度保持正确
                Cx[1] =  xi/dz;     Cz[1] =  (1-zi)/dx;
                Cx[3] =  (1-xi)/dz; Cz[3] = -(1-zi)/dx;
                Cx[5] = -xi/dz;     Cz[5] =  zi/dx;
                Cx[7] = -(1-xi)/dz; Cz[7] = -zi/dx;

                // 【修复核心 2】修正了 Z 族边 Cy 的正负号
                Cx[8] = -(1-xi)/dy; Cy[8] =  (1-yi)/dx;
                Cx[9] = -xi/dy;     Cy[9] = -(1-yi)/dx;
                Cx[10]=  xi/dy;     Cy[10]= -yi/dx;
                Cx[11]=  (1-xi)/dy; Cy[11]=  yi/dx;

                for(int i=0; i<12; i++) {
                    for(int j=0; j<12; j++) {
                        double curl_dot = Cx[i]*Cx[j] + Cy[i]*Cy[j] + Cz[i]*Cz[j];
                        double n_dot = Nx[i]*Nx[j] + Ny[i]*Ny[j] + Nz[i]*Nz[j];
                        std::complex<double> val = (oneOverM * curl_dot) + std::complex<double>(sigma * n_dot, 0.0);
                        Ke[i][j] += val * dV;
                    }
                }
            }
        }
    }
}

void femAssemble3D_Matrix(const HostMesh3D &mesh, double freq, const std::vector<double> &rho,
                          std::vector<std::complex<double>> &csrVal, 
                          std::vector<int> &csrColInd, 
                          std::vector<int> &csrRowPtr)
{
    if (mesh.NX <= 0 || mesh.NY <= 0 || mesh.NZ <= 0 || mesh.NE <= 0 || mesh.NEdges <= 0) {
        throw std::invalid_argument("mesh dimensions must be positive");
    }
    if (!std::isfinite(freq) || freq <= 0.0) {
        throw std::invalid_argument("frequency must be positive and finite");
    }
    if (rho.size() != static_cast<size_t>(mesh.NE)
        || mesh.ME_Edges.size() != static_cast<size_t>(12) * mesh.NE
        || mesh.hx.size() != static_cast<size_t>(mesh.NX)
        || mesh.hy.size() != static_cast<size_t>(mesh.NY)
        || mesh.hz.size() != static_cast<size_t>(mesh.NZ)) {
        throw std::invalid_argument("mesh and resistivity arrays have inconsistent sizes");
    }
    for (double value : rho) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument("resistivity values must be positive and finite");
        }
    }
    auto invalidWidth = [](double width) {
        return !std::isfinite(width) || width <= 0.0;
    };
    if (std::any_of(mesh.hx.begin(), mesh.hx.end(), invalidWidth)
        || std::any_of(mesh.hy.begin(), mesh.hy.end(), invalidWidth)
        || std::any_of(mesh.hz.begin(), mesh.hz.end(), invalidWidth)) {
        throw std::invalid_argument("mesh cell widths must be positive and finite");
    }
    if (mesh.NE > std::numeric_limits<int>::max() / 144) {
        throw std::overflow_error("element triplet count exceeds the supported range");
    }

    std::vector<ComplexTriple> triplets;
    triplets.reserve(static_cast<size_t>(mesh.NE) * 144);

    for (int h = 0; h < mesh.NE; h++) {
        double rh = rho[h]; 
        
        // 💡 提取该网格特有的长宽高
        int k = h / (mesh.NX * mesh.NY);
        int j = (h % (mesh.NX * mesh.NY)) / mesh.NX;
        int i = h % mesh.NX;
        double local_dx = mesh.hx[i];
        double local_dy = mesh.hy[j];
        double local_dz = mesh.hz[k];

        std::complex<double> Ke[12][12];
        computeLocalMatrix3D(local_dx, local_dy, local_dz, freq, rh, Ke);

        int globalEdges[12];
        for (int i = 0; i < 12; i++) {
            globalEdges[i] = mesh.ME_Edges[12 * h + i];
        }

        for (int i = 0; i < 12; i++) {
            int row = globalEdges[i];
            for (int j = 0; j < 12; j++) {
                int col = globalEdges[j];
                triplets.push_back({row, col, Ke[i][j]});
            }
        }
    }

    complexTripletToCSR3D(mesh.NEdges, triplets, csrRowPtr, csrColInd, csrVal);
}

void femBuild3D_CSRStructure(const HostMesh3D &mesh,
                             std::vector<int> &csrColInd,
                             std::vector<int> &csrRowPtr)
{
    if (mesh.NX <= 0 || mesh.NY <= 0 || mesh.NZ <= 0 || mesh.NE <= 0 || mesh.NEdges <= 0) {
        throw std::invalid_argument("mesh dimensions must be positive");
    }
    if (mesh.ME_Edges.size() != static_cast<size_t>(12) * mesh.NE) {
        throw std::invalid_argument("element-to-edge topology has an inconsistent size");
    }

    // Each edge belongs to at most four hexahedra. Build the short adjacency
    // list for each row directly instead of sorting tens of millions of global
    // complex triplets just to recover the sparsity pattern.
    std::vector<size_t> occurrenceCounts(mesh.NEdges, 0);
    for (int h = 0; h < mesh.NE; ++h) {
        for (int localRow = 0; localRow < 12; ++localRow) {
            int row = mesh.ME_Edges[12 * h + localRow];
            if (row < 0 || row >= mesh.NEdges) {
                throw std::out_of_range("element-to-edge topology contains an invalid edge index");
            }
            occurrenceCounts[row] += 12U;
        }
    }

    std::vector<size_t> occurrenceRowPtr(mesh.NEdges + 1, 0);
    for (int row = 0; row < mesh.NEdges; ++row) {
        occurrenceRowPtr[row + 1] = occurrenceRowPtr[row] + occurrenceCounts[row];
    }

    std::vector<int> occurrences(occurrenceRowPtr.back());
    std::vector<size_t> cursors = occurrenceRowPtr;
    for (int h = 0; h < mesh.NE; ++h) {
        const int* globalEdges = mesh.ME_Edges.data() + 12 * h;
        for (int localRow = 0; localRow < 12; ++localRow) {
            int row = globalEdges[localRow];
            size_t write = cursors[row];
            for (int localCol = 0; localCol < 12; ++localCol) {
                occurrences[write++] = globalEdges[localCol];
            }
            cursors[row] = write;
        }
    }

    csrRowPtr.assign(mesh.NEdges + 1, 0);
    size_t compactWrite = 0;
    for (int row = 0; row < mesh.NEdges; ++row) {
        auto begin = occurrences.begin() + occurrenceRowPtr[row];
        auto end = occurrences.begin() + occurrenceRowPtr[row + 1];
        std::sort(begin, end);

        if (compactWrite > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("CSR row pointers exceed the 32-bit index range");
        }
        csrRowPtr[row] = static_cast<int>(compactWrite);
        int previous = -1;
        for (auto it = begin; it != end; ++it) {
            if (*it != previous) {
                occurrences[compactWrite++] = *it;
                previous = *it;
            }
        }
    }
    if (compactWrite > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CSR nonzeros exceed the 32-bit index range");
    }
    csrRowPtr[mesh.NEdges] = static_cast<int>(compactWrite);
    occurrences.resize(compactWrite);
    csrColInd = std::move(occurrences);
}
