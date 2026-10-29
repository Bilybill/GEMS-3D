// src/gpuAssemble3d.cu
#include "../include/mesh3d.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

void checkKernelLaunch(const char* operation)
{
    checkCuda(cudaGetLastError(), operation);
}

} // namespace

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

struct EdgeBasisValue {
    double nx, ny, nz;
    double cx, cy, cz;
};

static __device__ __forceinline__ EdgeBasisValue evaluateEdgeBasis(
    int edge, double xi, double yi, double zi, double dx, double dy, double dz)
{
    EdgeBasisValue value = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    switch (edge) {
        case 0:
            value.nx = (1.0 - yi) * (1.0 - zi);
            value.cy = -(1.0 - yi) / dz;
            value.cz = (1.0 - zi) / dy;
            break;
        case 1:
            value.ny = xi * (1.0 - zi);
            value.cx = xi / dz;
            value.cz = (1.0 - zi) / dx;
            break;
        case 2:
            value.nx = yi * (1.0 - zi);
            value.cy = -yi / dz;
            value.cz = -(1.0 - zi) / dy;
            break;
        case 3:
            value.ny = (1.0 - xi) * (1.0 - zi);
            value.cx = (1.0 - xi) / dz;
            value.cz = -(1.0 - zi) / dx;
            break;
        case 4:
            value.nx = (1.0 - yi) * zi;
            value.cy = (1.0 - yi) / dz;
            value.cz = zi / dy;
            break;
        case 5:
            value.ny = xi * zi;
            value.cx = -xi / dz;
            value.cz = zi / dx;
            break;
        case 6:
            value.nx = yi * zi;
            value.cy = yi / dz;
            value.cz = -zi / dy;
            break;
        case 7:
            value.ny = (1.0 - xi) * zi;
            value.cx = -(1.0 - xi) / dz;
            value.cz = -zi / dx;
            break;
        case 8:
            value.nz = (1.0 - xi) * (1.0 - yi);
            value.cx = -(1.0 - xi) / dy;
            value.cy = (1.0 - yi) / dx;
            break;
        case 9:
            value.nz = xi * (1.0 - yi);
            value.cx = -xi / dy;
            value.cy = -(1.0 - yi) / dx;
            break;
        case 10:
            value.nz = xi * yi;
            value.cx = xi / dy;
            value.cy = -yi / dx;
            break;
        default:
            value.nz = (1.0 - xi) * yi;
            value.cx = (1.0 - xi) / dy;
            value.cy = yi / dx;
            break;
    }
    return value;
}

__global__ void assembleComponentMatricesKernel(
    int NE,
    int NX,
    int NY,
    const double* __restrict__ hx,
    const double* __restrict__ hy,
    const double* __restrict__ hz,
    const double* __restrict__ rho,
    const int* __restrict__ meEdges,
    const int* __restrict__ csrRowPtr,
    const int* __restrict__ csrColInd,
    double* __restrict__ stiffness,
    double* __restrict__ conductivityMass,
    int* __restrict__ invalidFlag)
{
    int element = blockIdx.x;
    int pair = threadIdx.x;
    if (element >= NE || pair >= 144) {
        return;
    }

    __shared__ double localDx;
    __shared__ double localDy;
    __shared__ double localDz;
    __shared__ double sigma;
    __shared__ EdgeBasisValue basisAtGaussPoints[12][8];
    if (pair == 0) {
        int planeSize = NX * NY;
        int k = element / planeSize;
        int remainder = element - k * planeSize;
        int j = remainder / NX;
        int i = remainder - j * NX;
        localDx = hx[i];
        localDy = hy[j];
        localDz = hz[k];
        sigma = 1.0 / rho[element];
    }
    __syncthreads();

    constexpr double invSqrt3 = 0.57735026918962576451;
    if (pair < 96) {
        int edge = pair / 8;
        int quadraturePoint = pair % 8;
        int ig = quadraturePoint / 4;
        int jg = (quadraturePoint / 2) % 2;
        int kg = quadraturePoint % 2;
        double xi = 0.5 + (ig == 0 ? -0.5 : 0.5) * invSqrt3;
        double yi = 0.5 + (jg == 0 ? -0.5 : 0.5) * invSqrt3;
        double zi = 0.5 + (kg == 0 ? -0.5 : 0.5) * invSqrt3;
        basisAtGaussPoints[edge][quadraturePoint] = evaluateEdgeBasis(
            edge, xi, yi, zi, localDx, localDy, localDz
        );
    }
    __syncthreads();

    int localRow = pair / 12;
    int localCol = pair % 12;
    int row = meEdges[12 * element + localRow];
    int col = meEdges[12 * element + localCol];
    int csrIndex = -1;
    for (int idx = csrRowPtr[row]; idx < csrRowPtr[row + 1]; ++idx) {
        if (csrColInd[idx] == col) {
            csrIndex = idx;
            break;
        }
    }
    if (csrIndex < 0) {
        atomicExch(invalidFlag, 1);
        return;
    }

    double weight = localDx * localDy * localDz * 0.125;
    double stiffnessValue = 0.0;
    double massValue = 0.0;

#pragma unroll
    for (int quadraturePoint = 0; quadraturePoint < 8; ++quadraturePoint) {
        EdgeBasisValue rowBasis = basisAtGaussPoints[localRow][quadraturePoint];
        EdgeBasisValue colBasis = basisAtGaussPoints[localCol][quadraturePoint];
        stiffnessValue += (
            rowBasis.cx * colBasis.cx
            + rowBasis.cy * colBasis.cy
            + rowBasis.cz * colBasis.cz
        ) * weight;
        massValue += sigma * (
            rowBasis.nx * colBasis.nx
            + rowBasis.ny * colBasis.ny
            + rowBasis.nz * colBasis.nz
        ) * weight;
    }

    if (stiffnessValue != 0.0) {
        atomicAddDoubleCompat(&stiffness[csrIndex], stiffnessValue);
    }
    if (massValue != 0.0) {
        atomicAddDoubleCompat(&conductivityMass[csrIndex], massValue);
    }
}

__global__ void combineComponentMatricesKernel(
    int nnz,
    double invOmegaMu,
    const double* __restrict__ stiffness,
    const double* __restrict__ conductivityMass,
    cuDoubleComplex* __restrict__ values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        values[idx] = make_cuDoubleComplex(
            conductivityMass[idx],
            -stiffness[idx] * invOmegaMu
        );
    }
}

extern "C" void gpu_assemble_component_matrices(
    int NE,
    int NX,
    int NY,
    int nnz,
    const double* d_hx,
    const double* d_hy,
    const double* d_hz,
    const double* d_rho,
    const int* d_meEdges,
    const int* d_csrRowPtr,
    const int* d_csrColInd,
    double* d_stiffness,
    double* d_conductivityMass,
    int* d_invalidFlag,
    cudaStream_t stream)
{
    if (NE <= 0 || NX <= 0 || NY <= 0 || nnz <= 0) {
        throw std::invalid_argument("GPU component assembly dimensions must be positive");
    }
    if (!d_hx || !d_hy || !d_hz || !d_rho || !d_meEdges
        || !d_csrRowPtr || !d_csrColInd || !d_stiffness
        || !d_conductivityMass || !d_invalidFlag) {
        throw std::invalid_argument("GPU component assembly received a null device pointer");
    }
    checkCuda(
        cudaMemsetAsync(d_stiffness, 0, static_cast<size_t>(nnz) * sizeof(double), stream),
        "clearing stiffness component"
    );
    checkCuda(
        cudaMemsetAsync(d_conductivityMass, 0, static_cast<size_t>(nnz) * sizeof(double), stream),
        "clearing conductivity component"
    );
    checkCuda(cudaMemsetAsync(d_invalidFlag, 0, sizeof(int), stream), "clearing assembly validation flag");
    assembleComponentMatricesKernel<<<NE, 144, 0, stream>>>(
        NE, NX, NY, d_hx, d_hy, d_hz, d_rho,
        d_meEdges, d_csrRowPtr, d_csrColInd,
        d_stiffness, d_conductivityMass, d_invalidFlag
    );
    checkKernelLaunch("launching component assembly kernel");
}

extern "C" void gpu_combine_component_matrices(
    int nnz,
    double freq,
    const double* d_stiffness,
    const double* d_conductivityMass,
    cuDoubleComplex* d_values,
    cudaStream_t stream)
{
    if (nnz <= 0 || !std::isfinite(freq) || freq <= 0.0) {
        throw std::invalid_argument("matrix combination requires a positive finite frequency and nonzero nnz");
    }
    if (!d_stiffness || !d_conductivityMass || !d_values) {
        throw std::invalid_argument("matrix combination received a null device pointer");
    }
    constexpr double mu = 4.0e-7 * M_PI;
    double invOmegaMu = 1.0 / (2.0 * M_PI * freq * mu);
    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    combineComponentMatricesKernel<<<blocks, threads, 0, stream>>>(
        nnz, invOmegaMu, d_stiffness, d_conductivityMass, d_values
    );
    checkKernelLaunch("launching matrix combination kernel");
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
                double dV = (dx * dy * dz) * gw * gw * gw;
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
    if (NE <= 0 || NX <= 0 || NY <= 0 || nnz <= 0
        || !std::isfinite(freq) || freq <= 0.0) {
        throw std::invalid_argument("GPU assembly dimensions and frequency must be positive and finite");
    }
    if (!d_hx || !d_hy || !d_hz || !d_rho || !d_ME_Edges
        || !d_csrRowPtr || !d_csrColInd || !d_csrVal) {
        throw std::invalid_argument("GPU assembly received a null device pointer");
    }
    // 每次计算新频率前，必须将矩阵值显存清零
    checkCuda(cudaMemset(d_csrVal, 0, nnz * sizeof(cuDoubleComplex)), "clearing assembled matrix");

    int threadsPerBlock = 256;
    int blocksPerGrid = (NE + threadsPerBlock - 1) / threadsPerBlock;

    assemble_csr_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NE, NX, NY, 
        d_hx, d_hy, d_hz, 
        d_rho, freq, 
        d_ME_Edges, 
        d_csrRowPtr, d_csrColInd, d_csrVal
    );
    checkKernelLaunch("launching legacy assembly kernel");
    checkCuda(cudaDeviceSynchronize(), "synchronizing legacy assembly kernel");
}
