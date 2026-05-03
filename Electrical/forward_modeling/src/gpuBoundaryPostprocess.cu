#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

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
    double* realAddr = reinterpret_cast<double*>(address);
    double* imagAddr = realAddr + 1;
    atomicAddDoubleCompat(realAddr, cuCreal(val));
    atomicAddDoubleCompat(imagAddr, cuCimag(val));
}

static __device__ __forceinline__ cuDoubleComplex complexAdd(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b));
}

static __device__ __forceinline__ cuDoubleComplex complexSub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b));
}

static __device__ __forceinline__ cuDoubleComplex complexMul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}

static __device__ __forceinline__ cuDoubleComplex complexScale(cuDoubleComplex a, double s) {
    return make_cuDoubleComplex(cuCreal(a) * s, cuCimag(a) * s);
}

static __device__ __forceinline__ cuDoubleComplex complexDiv(cuDoubleComplex a, cuDoubleComplex b) {
    double denom = cuCreal(b) * cuCreal(b) + cuCimag(b) * cuCimag(b);
    return make_cuDoubleComplex(
        (cuCreal(a) * cuCreal(b) + cuCimag(a) * cuCimag(b)) / denom,
        (cuCimag(a) * cuCreal(b) - cuCreal(a) * cuCimag(b)) / denom
    );
}

static __device__ __forceinline__ double complexAbsSq(cuDoubleComplex a) {
    return cuCreal(a) * cuCreal(a) + cuCimag(a) * cuCimag(a);
}

__global__ void computeBoundaryValuesKernel(
    int nBoundaryEdges,
    const int* boundaryEdgeFamilies,
    const int* boundaryEdgeZIndices,
    const double* zCoords,
    cuDoubleComplex k_num,
    int polarization,
    cuDoubleComplex* bcValues)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBoundaryEdges) {
        return;
    }

    int family = boundaryEdgeFamilies[idx];
    bool active = (family == 0 && polarization == 1) || (family == 1 && polarization == 2);
    if (!active) {
        bcValues[idx] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    double z = zCoords[boundaryEdgeZIndices[idx]];
    double realPart = cuCimag(k_num) * z;
    double imagPart = -cuCreal(k_num) * z;
    double amp = exp(realPart);
    bcValues[idx] = make_cuDoubleComplex(amp * cos(imagPart), amp * sin(imagPart));
}

__global__ void applyBoundaryCouplingsKernel(
    int nCouplings,
    const int* rows,
    const int* csrIndices,
    const int* boundaryPos,
    cuDoubleComplex* csrVal,
    const cuDoubleComplex* bcValues,
    cuDoubleComplex* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCouplings) {
        return;
    }

    int row = rows[idx];
    int csrIndex = csrIndices[idx];
    cuDoubleComplex contrib = complexMul(csrVal[csrIndex], bcValues[boundaryPos[idx]]);
    atomicAddComplex(&b[row], make_cuDoubleComplex(-cuCreal(contrib), -cuCimag(contrib)));
    csrVal[csrIndex] = make_cuDoubleComplex(0.0, 0.0);
}

__global__ void applyBoundaryRowsKernel(
    int nBoundaryRows,
    const int* rows,
    const int* rowStarts,
    const int* rowEnds,
    const int* diagIndices,
    const int* boundaryPos,
    cuDoubleComplex* csrVal,
    const cuDoubleComplex* bcValues,
    cuDoubleComplex* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBoundaryRows) {
        return;
    }

    int row = rows[idx];
    for (int csrIdx = rowStarts[idx]; csrIdx < rowEnds[idx]; ++csrIdx) {
        csrVal[csrIdx] = make_cuDoubleComplex(0.0, 0.0);
    }
    int diagIndex = diagIndices[idx];
    if (diagIndex >= 0) {
        csrVal[diagIndex] = make_cuDoubleComplex(1.0, 0.0);
    }
    b[row] = bcValues[boundaryPos[idx]];
}

__global__ void extractSurfaceFieldsKernel(
    int nxPad,
    int nyPad,
    int coreNx,
    int coreNy,
    int npad,
    double freq,
    const double* hx,
    const double* hy,
    const double* hz,
    const int* meEdges,
    const cuDoubleComplex* x,
    cuDoubleComplex* fields)
{
    int flatIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nStations = coreNx * coreNy;
    if (flatIdx >= nStations) {
        return;
    }

    int iCore = flatIdx % coreNx;
    int jCore = flatIdx / coreNx;
    int i = iCore + npad;
    int j = jCore + npad;
    int elemIdx = i + j * nxPad;

    double dx = hx[i];
    double dy = hy[j];
    double dz = hz[0];

    int e0 = meEdges[12 * elemIdx + 0];
    int e1 = meEdges[12 * elemIdx + 1];
    int e2 = meEdges[12 * elemIdx + 2];
    int e3 = meEdges[12 * elemIdx + 3];
    int e4 = meEdges[12 * elemIdx + 4];
    int e5 = meEdges[12 * elemIdx + 5];
    int e6 = meEdges[12 * elemIdx + 6];
    int e7 = meEdges[12 * elemIdx + 7];
    int e8 = meEdges[12 * elemIdx + 8];
    int e9 = meEdges[12 * elemIdx + 9];
    int e10 = meEdges[12 * elemIdx + 10];
    int e11 = meEdges[12 * elemIdx + 11];

    cuDoubleComplex exZ0 = complexScale(complexAdd(x[e0], x[e2]), 0.5);
    cuDoubleComplex eyZ0 = complexScale(complexAdd(x[e1], x[e3]), 0.5);
    cuDoubleComplex exZ1 = complexScale(complexAdd(x[e4], x[e6]), 0.5);
    cuDoubleComplex eyZ1 = complexScale(complexAdd(x[e5], x[e7]), 0.5);
    cuDoubleComplex ezY0 = complexScale(complexAdd(x[e8], x[e9]), 0.5);
    cuDoubleComplex ezY1 = complexScale(complexAdd(x[e10], x[e11]), 0.5);
    cuDoubleComplex ezX0 = complexScale(complexAdd(x[e8], x[e11]), 0.5);
    cuDoubleComplex ezX1 = complexScale(complexAdd(x[e9], x[e10]), 0.5);

    cuDoubleComplex exCenter = complexScale(complexAdd(exZ0, exZ1), 0.5);
    cuDoubleComplex eyCenter = complexScale(complexAdd(eyZ0, eyZ1), 0.5);

    cuDoubleComplex dEyDz = complexScale(complexSub(eyZ1, eyZ0), 1.0 / dz);
    cuDoubleComplex dExDz = complexScale(complexSub(exZ1, exZ0), 1.0 / dz);
    cuDoubleComplex dEzDy = complexScale(complexSub(ezY1, ezY0), 1.0 / dy);
    cuDoubleComplex dEzDx = complexScale(complexSub(ezX1, ezX0), 1.0 / dx);

    double mu = 4e-7 * M_PI;
    cuDoubleComplex negIwMu = make_cuDoubleComplex(0.0, -2.0 * M_PI * freq * mu);
    cuDoubleComplex hxField = complexDiv(complexSub(dEzDy, dEyDz), negIwMu);
    cuDoubleComplex hyField = complexDiv(complexSub(dExDz, dEzDx), negIwMu);

    fields[4 * flatIdx + 0] = exCenter;
    fields[4 * flatIdx + 1] = eyCenter;
    fields[4 * flatIdx + 2] = hxField;
    fields[4 * flatIdx + 3] = hyField;
}

__global__ void computeResponsesKernel(
    int nx,
    int ny,
    double freq,
    const cuDoubleComplex* fieldsPol1,
    const cuDoubleComplex* fieldsPol2,
    double* appResSlice,
    double* phaseSlice)
{
    int flatIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nStations = nx * ny;
    if (flatIdx >= nStations) {
        return;
    }

    // MTForward3D's raw station order is flatIdx = x + y * nx (x-fastest).
    // GMESDataset exposes tensors as [freq, x, y, component], whose contiguous
    // memory order is tensorIdx = x * ny + y. Convert explicitly here so the
    // returned tensor matches the documented [NX, NY] indexing convention.
    int xIdx = flatIdx % nx;
    int yIdx = flatIdx / nx;
    int tensorIdx = xIdx * ny + yIdx;

    cuDoubleComplex f1Ex = fieldsPol1[4 * flatIdx + 0];
    cuDoubleComplex f1Ey = fieldsPol1[4 * flatIdx + 1];
    cuDoubleComplex f1Hx = fieldsPol1[4 * flatIdx + 2];
    cuDoubleComplex f1Hy = fieldsPol1[4 * flatIdx + 3];

    cuDoubleComplex f2Ex = fieldsPol2[4 * flatIdx + 0];
    cuDoubleComplex f2Ey = fieldsPol2[4 * flatIdx + 1];
    cuDoubleComplex f2Hx = fieldsPol2[4 * flatIdx + 2];
    cuDoubleComplex f2Hy = fieldsPol2[4 * flatIdx + 3];

    cuDoubleComplex detH = complexSub(complexMul(f1Hx, f2Hy), complexMul(f2Hx, f1Hy));
    double rhoXY = 0.0;
    double rhoYX = 0.0;
    double phiXY = 0.0;
    double phiYX = 0.0;

    if (complexAbsSq(detH) > 1e-30) {
        cuDoubleComplex zxy = complexDiv(
            complexSub(complexMul(f2Ex, f1Hx), complexMul(f1Ex, f2Hx)),
            detH
        );
        cuDoubleComplex zyx = complexDiv(
            complexSub(complexMul(f1Ey, f2Hy), complexMul(f2Ey, f1Hy)),
            detH
        );

        double mu = 4e-7 * M_PI;
        rhoXY = complexAbsSq(zxy) / (2.0 * M_PI * freq * mu);
        rhoYX = complexAbsSq(zyx) / (2.0 * M_PI * freq * mu);
        phiXY = atan2(cuCimag(zxy), cuCreal(zxy)) * 180.0 / M_PI;
        phiYX = atan2(cuCimag(zyx), cuCreal(zyx)) * 180.0 / M_PI;
    }

    appResSlice[2 * tensorIdx + 0] = rhoXY;
    appResSlice[2 * tensorIdx + 1] = rhoYX;
    phaseSlice[2 * tensorIdx + 0] = phiXY;
    phaseSlice[2 * tensorIdx + 1] = phiYX;
}

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
    cuDoubleComplex* d_bcValues)
{
    int threads = 256;
    if (boundaryEdgeCount > 0) {
        int blocks = (boundaryEdgeCount + threads - 1) / threads;
        computeBoundaryValuesKernel<<<blocks, threads>>>(
            boundaryEdgeCount,
            d_boundaryEdgeFamilies,
            d_boundaryEdgeZIndices,
            d_zCoords,
            k_num,
            polarization,
            d_bcValues
        );
    }

    if (couplingCount > 0) {
        int blocks = (couplingCount + threads - 1) / threads;
        applyBoundaryCouplingsKernel<<<blocks, threads>>>(
            couplingCount,
            d_couplingRows,
            d_couplingCsrIndices,
            d_couplingBoundaryPos,
            d_csrVal,
            d_bcValues,
            d_b
        );
    }

    if (boundaryRowCount > 0) {
        int blocks = (boundaryRowCount + threads - 1) / threads;
        applyBoundaryRowsKernel<<<blocks, threads>>>(
            boundaryRowCount,
            d_boundaryRowRows,
            d_boundaryRowStarts,
            d_boundaryRowEnds,
            d_boundaryRowDiagIndices,
            d_boundaryRowBoundaryPos,
            d_csrVal,
            d_bcValues,
            d_b
        );
    }
    cudaDeviceSynchronize();
}

extern "C" void gpu_extract_surface_fields_2d(
    int nxPad,
    int nyPad,
    int coreNx,
    int coreNy,
    int npad,
    double freq,
    const double* d_hx,
    const double* d_hy,
    const double* d_hz,
    const int* d_meEdges,
    const cuDoubleComplex* d_x,
    cuDoubleComplex* d_fields)
{
    int nStations = coreNx * coreNy;
    int threads = 256;
    int blocks = (nStations + threads - 1) / threads;
    extractSurfaceFieldsKernel<<<blocks, threads>>>(
        nxPad, nyPad, coreNx, coreNy, npad, freq, d_hx, d_hy, d_hz, d_meEdges, d_x, d_fields
    );
    cudaDeviceSynchronize();
}

extern "C" void gpu_compute_mt_responses(
    int nx,
    int ny,
    double freq,
    const cuDoubleComplex* d_fields_pol1,
    const cuDoubleComplex* d_fields_pol2,
    double* d_app_res_slice,
    double* d_phase_slice)
{
    int nStations = nx * ny;
    int threads = 256;
    int blocks = (nStations + threads - 1) / threads;
    computeResponsesKernel<<<blocks, threads>>>(
        nx, ny, freq, d_fields_pol1, d_fields_pol2, d_app_res_slice, d_phase_slice
    );
    cudaDeviceSynchronize();
}
