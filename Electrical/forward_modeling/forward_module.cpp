#include <torch/extension.h>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <ATen/TensorIndexing.h>

#include "include/mesh3d.h"
#include "include/femAssemble3d.h"
#include "include/boundaryConditions3d.h"
#include "include/gpuIterativeSolver.h"

extern "C" void gpu_femAssemble3D_Matrix(
    int NE, int NX, int NY,
    const double* d_hx, const double* d_hy, const double* d_hz,
    const double* d_rho, double freq,
    const int* d_ME_Edges,
    const int* d_csrRowPtr, const int* d_csrColInd, 
    cuDoubleComplex* d_csrVal, int nnz);

extern "C" void gpu_extract_surface_fields_2d(
    int nxPad, int nyPad, int coreNx, int coreNy, int npad, double freq,
    const double* d_hx, const double* d_hy, const double* d_hz,
    const int* d_meEdges, const cuDoubleComplex* d_x, cuDoubleComplex* d_fields);

extern "C" void gpu_compute_mt_responses(
    int nx, int ny, double freq,
    const cuDoubleComplex* d_fields_pol1,
    const cuDoubleComplex* d_fields_pol2,
    double* d_app_res_slice,
    double* d_phase_slice);

namespace {

double estimate_boundary_background_rho(const torch::Tensor& rho_tensor) {
    using namespace torch::indexing;

    const auto nx = rho_tensor.size(0);
    const auto ny = rho_tensor.size(1);
    const auto nz = rho_tensor.size(2);

    auto mask = torch::zeros({nx, ny, nz}, rho_tensor.options().dtype(torch::kBool));
    mask.index_put_({0, Slice(), Slice()}, true);
    mask.index_put_({nx - 1, Slice(), Slice()}, true);
    mask.index_put_({Slice(), 0, Slice()}, true);
    mask.index_put_({Slice(), ny - 1, Slice()}, true);
    mask.index_put_({Slice(), Slice(), 0}, true);
    mask.index_put_({Slice(), Slice(), nz - 1}, true);
    return rho_tensor.masked_select(mask).mean().item<double>();
}

torch::Tensor build_padded_rho_tensor(const torch::Tensor& rho_tensor, int npad) {
    auto index_options = torch::TensorOptions().device(rho_tensor.device()).dtype(torch::kLong);

    const auto nx = rho_tensor.size(0);
    const auto ny = rho_tensor.size(1);
    const auto nz = rho_tensor.size(2);

    auto x_idx = torch::clamp(torch::arange(nx + 2 * npad, index_options) - npad, 0, nx - 1);
    auto y_idx = torch::clamp(torch::arange(ny + 2 * npad, index_options) - npad, 0, ny - 1);
    auto z_idx = torch::clamp(torch::arange(nz + npad, index_options), 0, nz - 1);

    return rho_tensor.index_select(0, x_idx)
                     .index_select(1, y_idx)
                     .index_select(2, z_idx)
                     .contiguous();
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> compute_mt_3d(torch::Tensor rho_tensor, double dx, double dy, double dz, std::vector<double> freqs) {
    TORCH_CHECK(rho_tensor.dim() == 3, "rho_tensor must be 3D with shape (NX, NY, NZ)");
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    TORCH_CHECK(
        cuda_status == cudaSuccess && device_count > 0,
        "mt_forward_cuda requires a working CUDA runtime and at least one visible GPU. "
        "cudaGetDeviceCount failed with: ", cudaGetErrorString(cuda_status)
    );

    auto rho_work = rho_tensor;
    if (rho_work.scalar_type() != torch::kFloat64) {
        rho_work = rho_work.to(torch::kFloat64);
    }
    if (!rho_work.is_contiguous()) {
        rho_work = rho_work.contiguous();
    }

    if (rho_work.is_cuda()) {
        cudaSetDevice(rho_work.get_device());
    }
    
    int NX = rho_work.size(0);
    int NY = rho_work.size(1);
    int NZ = rho_work.size(2);

    double bg_rho_for_bc = estimate_boundary_background_rho(rho_work);

    int npad = 10;
    double alpha = 1.4;
    
    int NX_pad = NX + 2 * npad;
    int NY_pad = NY + 2 * npad;
    int NZ_pad = NZ + npad; 
    
    std::vector<double> hx(NX_pad), hy(NY_pad), hz(NZ_pad);
    for(int i=0; i<npad; i++) hx[npad - 1 - i] = dx * std::pow(alpha, i+1); 
    for(int i=0; i<NX; i++) hx[npad + i] = dx;                              
    for(int i=0; i<npad; i++) hx[npad + NX + i] = dx * std::pow(alpha, i+1);
    
    for(int i=0; i<npad; i++) hy[npad - 1 - i] = dy * std::pow(alpha, i+1); 
    for(int i=0; i<NY; i++) hy[npad + i] = dy;                              
    for(int i=0; i<npad; i++) hy[npad + NY + i] = dy * std::pow(alpha, i+1);

    for(int i=0; i<NZ; i++) hz[i] = dz;                                     
    for(int i=0; i<npad; i++) hz[NZ + i] = dz * std::pow(alpha, i+1);       

    auto rho_pad_tensor = build_padded_rho_tensor(rho_work, npad);
    // The FEM/CUDA kernels expect the flattened resistivity ordering used by the
    // original MTForward3D code: idx = x + y * NX + z * NX * NY (x-fastest).
    // A contiguous torch tensor with logical shape (NX, NY, NZ) is laid out with
    // z-fastest storage, so we materialize a (NZ, NY, NX)-contiguous view before
    // exposing its raw storage pointer to the solver.
    auto rho_pad_linear_tensor = rho_pad_tensor.permute({2, 1, 0}).contiguous();

    auto rho_pad_linear_cpu = rho_pad_linear_tensor.is_cuda()
        ? rho_pad_linear_tensor.to(torch::kCPU)
        : rho_pad_linear_tensor;
    if (!rho_pad_linear_cpu.is_contiguous()) {
        rho_pad_linear_cpu = rho_pad_linear_cpu.contiguous();
    }
    std::vector<double> rho_pad_host(
        rho_pad_linear_cpu.data_ptr<double>(),
        rho_pad_linear_cpu.data_ptr<double>() + rho_pad_linear_cpu.numel()
    );

    HostMesh3D mesh = generateMesh3D(NX_pad, NY_pad, NZ_pad, hx, hy, hz);
    int NDoF = mesh.NEdges;
    
    if (freqs.empty()) {
        double skin_depth_min = dz * 2.0;
        double total_depth = NZ * dz;
        double skin_depth_max = total_depth / 1.5;
        double f_max = bg_rho_for_bc * std::pow(503.0 / skin_depth_min, 2.0);
        double f_min = bg_rho_for_bc * std::pow(503.0 / skin_depth_max, 2.0);

        std::vector<double> phoenix_coeffs = {8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0};
        int max_power = std::ceil(std::log10(f_max));
        int min_power = std::floor(std::log10(f_min));

        for (int p = max_power; p >= min_power; --p) {
            double power_of_10 = std::pow(10.0, p);
            for (double coeff : phoenix_coeffs) {
                double current_f = coeff * power_of_10;
                if (current_f <= f_max && current_f >= f_min) {
                    freqs.push_back(current_f);
                }
            }
        }

        if (freqs.empty()) {
            freqs.push_back(f_max);
            freqs.push_back(f_min);
        }
    }

    int n_freqs = (int)freqs.size();

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(rho_work.device());
    // Keep GMESDataset's repository-wide convention: [n_freq, NX, NY, 2].
    torch::Tensor app_res = torch::empty({n_freqs, NX, NY, 2}, options);
    torch::Tensor phase = torch::empty({n_freqs, NX, NY, 2}, options);
    
    double* res_ptr = app_res.data_ptr<double>();
    double* phs_ptr = phase.data_ptr<double>();

    std::cout << "3D MT Forward Modeling Started (Right-Preconditioned BiCGStab)...\n";
    std::cout << "Mesh: " << NX << "x" << NY << "x" << NZ << ", Cell: " << dx << "x" << dy << "x" << dz << " m\n";
    std::cout << "-> Auto-calculated Background Rho: " << bg_rho_for_bc << " Ohm-m\n";
    std::cout << "-> Auto-Padding Applied: Mesh expanded to " << NX_pad << "x" << NY_pad << "x" << NZ_pad << std::endl;
    std::cout << "   Range: [" << freqs.front() << " Hz  ...  " << freqs.back() << " Hz], Points = " << n_freqs << "\n";

    // Reuse MTForward3D's original assembled sparsity pattern so the solver and
    // boundary-condition plans see the exact same CSR structure as the reference
    // implementation.
    std::vector<std::complex<double>> dummy_csrVal;
    std::vector<int> h_csrColInd, h_csrRowPtr;
    femAssemble3D_Matrix(mesh, 1.0, rho_pad_host, dummy_csrVal, h_csrColInd, h_csrRowPtr);
    int nnz = dummy_csrVal.size();

    // ==========================================
    // 分配 GPU 长驻显存：坐标、拓扑与 CSR 结构
    // ==========================================
    double *d_hx, *d_hy, *d_hz;
    double *d_rho_owned = nullptr;
    const double* d_rho = nullptr;
    int *d_ME_Edges;
    cuDoubleComplex *d_csrVal_base, *d_x_pol1, *d_x_pol2, *d_fields_pol1, *d_fields_pol2;
    double *d_app_res, *d_phase;

    cudaMalloc(reinterpret_cast<void**>(&d_hx), mesh.NX * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_hy), mesh.NY * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_hz), mesh.NZ * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_ME_Edges), mesh.ME_Edges.size() * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_csrVal_base), nnz * sizeof(cuDoubleComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_x_pol1), NDoF * sizeof(cuDoubleComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_x_pol2), NDoF * sizeof(cuDoubleComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_fields_pol1), static_cast<size_t>(NX) * NY * 4 * sizeof(cuDoubleComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_fields_pol2), static_cast<size_t>(NX) * NY * 4 * sizeof(cuDoubleComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_app_res), static_cast<size_t>(n_freqs) * NX * NY * 2 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_phase), static_cast<size_t>(n_freqs) * NX * NY * 2 * sizeof(double));

    cudaMemcpy(d_hx, mesh.hx.data(), mesh.NX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hy, mesh.hy.data(), mesh.NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hz, mesh.hz.data(), mesh.NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ME_Edges, mesh.ME_Edges.data(), mesh.ME_Edges.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_x_pol1, 0, NDoF * sizeof(cuDoubleComplex));
    cudaMemset(d_x_pol2, 0, NDoF * sizeof(cuDoubleComplex));

    if (rho_pad_linear_tensor.is_cuda()) {
        d_rho = rho_pad_linear_tensor.data_ptr<double>();
    } else {
        cudaMalloc(reinterpret_cast<void**>(&d_rho_owned), mesh.NE * sizeof(double));
        cudaMemcpy(d_rho_owned, rho_pad_host.data(), mesh.NE * sizeof(double), cudaMemcpyHostToDevice);
        d_rho = d_rho_owned;
    }

    GpuBiCGStabContext* solverContext = create_gpu_bicgstab_context(NDoF, nnz);
    initialize_gpu_bicgstab_structure(solverContext, h_csrRowPtr, h_csrColInd);
    const int* d_csrRowPtr = get_gpu_bicgstab_row_ptr_device(solverContext);
    const int* d_csrColInd = get_gpu_bicgstab_col_ind_device(solverContext);
    cuDoubleComplex* d_csrVal_work = get_gpu_bicgstab_matrix_values_device(solverContext);
    cuDoubleComplex* d_b_work = get_gpu_bicgstab_rhs_device(solverContext);
    BoundaryConditionPlan* boundaryPlan = createBoundaryConditionPlan3D(mesh, h_csrColInd, h_csrRowPtr);

    for (int ff = 0; ff < n_freqs; ff++) {
        double freq = freqs[ff];
        std::cout << "\n=== Processing Frequency " << ff + 1 << "/" << n_freqs << ": " << freq << " Hz ===" << std::endl;

        // 调用 GPU 组装
        gpu_femAssemble3D_Matrix(mesh.NE, mesh.NX, mesh.NY, d_hx, d_hy, d_hz, d_rho, freq,
                                 d_ME_Edges, d_csrRowPtr, d_csrColInd, d_csrVal_base, nnz);

        cudaMemcpy(d_csrVal_work, d_csrVal_base, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        applyBoundaryConditions3DDevice(freq, bg_rho_for_bc, d_csrVal_work, d_b_work, 1, boundaryPlan);
        std::cout << "  -> Solving Polarization 1 (Ex) using BiCGStab..." << std::endl;
        SolverStats pol1Stats;
        gpu_complex_solver_bicgstab_device(solverContext, d_x_pol1, 1e-7, 50000, &pol1Stats);
        std::cout << "     Polarization 1 stats: iterations=" << pol1Stats.iterations
                  << ", residual=" << pol1Stats.finalResidual << std::endl;
        gpu_extract_surface_fields_2d(mesh.NX, mesh.NY, NX, NY, npad, freq, d_hx, d_hy, d_hz, d_ME_Edges, d_x_pol1, d_fields_pol1);

        cudaMemcpy(d_csrVal_work, d_csrVal_base, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        applyBoundaryConditions3DDevice(freq, bg_rho_for_bc, d_csrVal_work, d_b_work, 2, boundaryPlan);
        std::cout << "  -> Solving Polarization 2 (Ey) using BiCGStab..." << std::endl;
        SolverStats pol2Stats;
        gpu_complex_solver_bicgstab_device(solverContext, d_x_pol2, 1e-7, 50000, &pol2Stats);
        std::cout << "     Polarization 2 stats: iterations=" << pol2Stats.iterations
                  << ", residual=" << pol2Stats.finalResidual << std::endl;
        gpu_extract_surface_fields_2d(mesh.NX, mesh.NY, NX, NY, npad, freq, d_hx, d_hy, d_hz, d_ME_Edges, d_x_pol2, d_fields_pol2);

        gpu_compute_mt_responses(
            NX, NY, freq,
            d_fields_pol1,
            d_fields_pol2,
            d_app_res + static_cast<size_t>(ff) * NX * NY * 2,
            d_phase + static_cast<size_t>(ff) * NX * NY * 2
        );
    }

    auto output_copy_kind = app_res.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    cudaMemcpy(res_ptr, d_app_res, static_cast<size_t>(n_freqs) * NX * NY * 2 * sizeof(double), output_copy_kind);
    cudaMemcpy(phs_ptr, d_phase, static_cast<size_t>(n_freqs) * NX * NY * 2 * sizeof(double), output_copy_kind);

    cudaFree(d_hx);
    cudaFree(d_hy);
    cudaFree(d_hz);
    cudaFree(d_rho_owned);
    cudaFree(d_ME_Edges);
    cudaFree(d_csrVal_base);
    cudaFree(d_x_pol1);
    cudaFree(d_x_pol2);
    cudaFree(d_fields_pol1);
    cudaFree(d_fields_pol2);
    cudaFree(d_app_res);
    cudaFree(d_phase);
    destroyBoundaryConditionPlan3D(boundaryPlan);
    destroy_gpu_bicgstab_context(solverContext);

    return std::make_tuple(app_res, phase);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_mt_3d", &compute_mt_3d, "MT 3D Forward Modeling");
}
