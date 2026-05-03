// src/gpuAssemble3d.cu
#include "../include/mesh3d.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>

// 1. 针对 double 类型的复数原子加法 (CUDA 原生只有 float 版本的复数原子加)
static __device__ __forceinline__ double atomicAddDoubleCompat(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* addressAsUll = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addressAsUll;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            addressAsUll,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed))
        );
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

static __device__ __forceinline__ void atomicAddComplex(cuDoubleComplex* address, cuDoubleComplex val) {
    double* realAddr = (double*)address;
    double* imagAddr = realAddr + 1;
    atomicAddDoubleCompat(realAddr, cuCreal(val));
    atomicAddDoubleCompat(imagAddr, cuCimag(val));
}

// 2. 将 CPU 的 computeLocalMatrix3D 移植为设备函数
__device__ void computeLocalMatrix3D_device(double dx, double dy, double dz, double freq, double rho_val, cuDoubleComplex Ke[12][12]) {
    double mu = 4e-7 * M_PI;
    double w = 2.0 * M_PI * freq;
    
    // oneOverM = 1.0 / (i * w * mu) = -i / (w * mu)
    cuDoubleComplex oneOverM = make_cuDoubleComplex(0.0, -1.0 / (w * mu));
    double sigma = 1.0 / rho_val;

    for(int i = 0; i < 12; i++) {
        for(int j = 0; j < 12; j++) {
            Ke[i][j] = make_cuDoubleComplex(0.0, 0.0);
        }
    }

    double gp[2] = {0.5 - 0.5/sqrt(3.0), 0.5 + 0.5/sqrt(3.0)};
    double gw = 0.5;

    for(int ig = 0; ig < 2; ig++) {
        for(int jg = 0; jg < 2; jg++) {
            for(int kg = 0; kg < 2; kg++) {
                double x = gp[ig] * dx, y = gp[jg] * dy, z = gp[kg] * dz;
                double dV = (dx * dy * dz) * (gw * 2) * (gw * 2) * (gw * 2);
                double xi = x/dx, yi = y/dy, zi = z/dz;
                
                double Nx[12]={0}, Ny[12]={0}, Nz[12]={0};
                Nx[0] = (1-yi)*(1-zi); Nx[2] = yi*(1-zi); Nx[4] = (1-yi)*zi; Nx[6] = yi*zi;
                Ny[1] = xi*(1-zi); Ny[3] = (1-xi)*(1-zi); Ny[5] = xi*zi; Ny[7] = (1-xi)*zi;
                Nz[8] = (1-xi)*(1-yi); Nz[9] = xi*(1-yi); Nz[10] = xi*yi; Nz[11] = (1-xi)*yi;

                double Cx[12]={0}, Cy[12]={0}, Cz[12]={0};
                
                Cy[0] = -(1-yi)/dz; Cz[0] =  (1-zi)/dy;
                Cy[2] = -yi/dz;     Cz[2] = -(1-zi)/dy;
                Cy[4] =  (1-yi)/dz; Cz[4] =  zi/dy;
                Cy[6] =  yi/dz;     Cz[6] = -zi/dy;

                Cx[1] =  xi/dz;     Cz[1] =  (1-zi)/dx;
                Cx[3] =  (1-xi)/dz; Cz[3] = -(1-zi)/dx;
                Cx[5] = -xi/dz;     Cz[5] =  zi/dx;
                Cx[7] = -(1-xi)/dz; Cz[7] = -zi/dx;

                Cx[8] = -(1-xi)/dy; Cy[8] =  (1-yi)/dx;
                Cx[9] = -xi/dy;     Cy[9] = -(1-yi)/dx;
                Cx[10]=  xi/dy;     Cy[10]= -yi/dx;
                Cx[11]=  (1-xi)/dy; Cy[11]=  yi/dx;

                for(int i = 0; i < 12; i++) {
                    for(int j = 0; j < 12; j++) {
                        double curl_dot = Cx[i]*Cx[j] + Cy[i]*Cy[j] + Cz[i]*Cz[j];
                        double n_dot = Nx[i]*Nx[j] + Ny[i]*Ny[j] + Nz[i]*Nz[j];
                        
                        cuDoubleComplex term1 = make_cuDoubleComplex(cuCreal(oneOverM) * curl_dot, cuCimag(oneOverM) * curl_dot);
                        cuDoubleComplex term2 = make_cuDoubleComplex(sigma * n_dot, 0.0);
                        
                        cuDoubleComplex val = make_cuDoubleComplex(
                            (cuCreal(term1) + cuCreal(term2)) * dV, 
                            (cuCimag(term1) + cuCimag(term2)) * dV
                        );
                        
                        Ke[i][j] = make_cuDoubleComplex(cuCreal(Ke[i][j]) + cuCreal(val), cuCimag(Ke[i][j]) + cuCimag(val));
                    }
                }
            }
        }
    }
}

// 3. 全局组装 Kernel (每个线程负责一个六面体单元)
__global__ void assemble_csr_kernel(
    int NE, int NX, int NY,
    const double* __restrict__ hx,
    const double* __restrict__ hy,
    const double* __restrict__ hz,
    const double* __restrict__ rho,
    double freq,
    const int* __restrict__ ME_Edges,
    const int* __restrict__ csrRowPtr,
    const int* __restrict__ csrColInd,
    cuDoubleComplex* __restrict__ csrVal)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= NE) return;

    // 解析当前单元的 3D 索引
    int k = h / (NX * NY);
    int j = (h % (NX * NY)) / NX;
    int i = h % NX;

    double local_dx = hx[i];
    double local_dy = hy[j];
    double local_dz = hz[k];
    double rh = rho[h];

    // 计算局域刚度矩阵
    cuDoubleComplex Ke[12][12];
    computeLocalMatrix3D_device(local_dx, local_dy, local_dz, freq, rh, Ke);

    // 获取当前单元的 12 条全局边索引
    int globalEdges[12];
    for (int e = 0; e < 12; e++) {
        globalEdges[e] = ME_Edges[12 * h + e];
    }

    // 将局部刚度矩阵累加到全局 CSR 中
    for (int r = 0; r < 12; r++) {
        int row = globalEdges[r];
        int start_idx = csrRowPtr[row];
        int end_idx = csrRowPtr[row + 1];

        for (int c = 0; c < 12; c++) {
            int col = globalEdges[c];
            cuDoubleComplex val = Ke[r][c];

            // ⚠️ 极其核心：在稀疏行内寻找对应的列索引并进行原子加法
            // 因为边缘元最高度数较小(通常10~20)，这里的线性查找非常快，且全部命中 L1 Cache
            for (int idx = start_idx; idx < end_idx; idx++) {
                if (csrColInd[idx] == col) {
                    atomicAddComplex(&csrVal[idx], val);
                    break;
                }
            }
        }
    }
}

// 4. 暴露给 C++ 的 Host 端封装函数
extern "C" void gpu_femAssemble3D_Matrix(
    int NE, int NX, int NY,
    const double* d_hx, const double* d_hy, const double* d_hz,
    const double* d_rho, double freq,
    const int* d_ME_Edges,
    const int* d_csrRowPtr, const int* d_csrColInd, 
    cuDoubleComplex* d_csrVal, int nnz)
{
    // 每次计算新频率前，必须将矩阵值显存清零
    cudaMemset(d_csrVal, 0, nnz * sizeof(cuDoubleComplex));

    int threadsPerBlock = 256;
    int blocksPerGrid = (NE + threadsPerBlock - 1) / threadsPerBlock;

    assemble_csr_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NE, NX, NY, 
        d_hx, d_hy, d_hz, 
        d_rho, freq, 
        d_ME_Edges, 
        d_csrRowPtr, d_csrColInd, d_csrVal
    );
    cudaDeviceSynchronize();
}
