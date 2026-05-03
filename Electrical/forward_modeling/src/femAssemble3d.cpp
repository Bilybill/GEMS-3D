// src/femAssemble3d.cpp
#include "../include/femAssemble3d.h"
#include <iostream>
#include <algorithm>
#include <cmath>

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
                double dV = (dx * dy * dz) * (gw * 2) * (gw * 2) * (gw * 2); 
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
                std::complex<double> val = Ke[i][j];
                if (std::abs(val) > 1e-20) {
                    triplets.push_back({row, col, val});
                }
            }
        }
    }

    complexTripletToCSR3D(mesh.NEdges, triplets, csrRowPtr, csrColInd, csrVal);
}

void femBuild3D_CSRStructure(const HostMesh3D &mesh,
                             std::vector<int> &csrColInd,
                             std::vector<int> &csrRowPtr)
{
    std::vector<ComplexTriple> triplets;
    triplets.reserve(static_cast<size_t>(mesh.NE) * 144);

    for (int h = 0; h < mesh.NE; ++h) {
        int globalEdges[12];
        for (int e = 0; e < 12; ++e) {
            globalEdges[e] = mesh.ME_Edges[12 * h + e];
        }

        for (int i = 0; i < 12; ++i) {
            int row = globalEdges[i];
            for (int j = 0; j < 12; ++j) {
                int col = globalEdges[j];
                triplets.push_back({row, col, std::complex<double>(1.0, 0.0)});
            }
        }
    }

    std::vector<std::complex<double>> dummyVals;
    complexTripletToCSR3D(mesh.NEdges, triplets, csrRowPtr, csrColInd, dummyVals);
}
