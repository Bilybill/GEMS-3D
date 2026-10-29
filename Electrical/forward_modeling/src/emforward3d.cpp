// src/emforward3d.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "../include/mesh3d.h"
#include "../include/femAssemble3d.h"
#include "../include/boundaryConditions3d.h"
#include "../include/gpuIterativeSolver.h"

extern "C" void gpu_assemble_component_matrices(
    int NE, int NX, int NY, int nnz,
    const double* d_hx, const double* d_hy, const double* d_hz,
    const double* d_rho, const int* d_meEdges,
    const int* d_csrRowPtr, const int* d_csrColInd,
    double* d_stiffness, double* d_conductivityMass,
    int* d_invalidFlag, cudaStream_t stream);

extern "C" void gpu_combine_component_matrices(
    int nnz, double freq,
    const double* d_stiffness, const double* d_conductivityMass,
    cuDoubleComplex* d_values, cudaStream_t stream);

extern "C" void gpu_extract_surface_fields_2d(
    int nxPad, int nyPad, int coreNx, int coreNy, int npad,
    int surfaceLayer, double freq,
    const double* d_hx, const double* d_hy, const double* d_hz,
    const int* d_meEdges, const cuDoubleComplex* d_x,
    cuDoubleComplex* d_fields, cudaStream_t stream);

extern "C" void gpu_compute_mt_responses(
    int nx, int ny, double freq,
    const cuDoubleComplex* d_fields_pol1,
    const cuDoubleComplex* d_fields_pol2,
    double* d_app_res_slice, double* d_phase_slice,
    cudaStream_t stream);

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

void checkLastCudaError(const char* operation) {
    checkCuda(cudaGetLastError(), operation);
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count > 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                      "allocating CUDA buffer");
        }
    }

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

struct SolverContextDeleter {
    void operator()(GpuBiCGStabContext* context) const {
        destroy_gpu_bicgstab_context(context);
    }
};

struct BoundaryPlanDeleter {
    void operator()(BoundaryConditionPlan* plan) const {
        destroyBoundaryConditionPlan3D(plan);
    }
};

int checkedMeshProduct(int nx, int ny, int nz, const char* label) {
    long long product = static_cast<long long>(nx) * ny * nz;
    if (product <= 0 || product > INT_MAX) {
        throw std::overflow_error(std::string(label) + " exceeds 32-bit mesh capacity");
    }
    return static_cast<int>(product);
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <rhofilename> <NX> <NY> <NZ> <dx> <dy> <dz> [f_min] [f_max] [bg_rho] [padding_layers] [padding_alpha]\n";
        return 1;
    }

    try {

    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count <= 0) {
        std::cerr << "Error: emforward3d requires a working CUDA runtime and at least one visible GPU. "
                  << "cudaGetDeviceCount failed with: " << cudaGetErrorString(cuda_status) << "\n";
        return 1;
    }
    
    std::string rhofilename = argv[1];
    int NX = std::atoi(argv[2]);
    int NY = std::atoi(argv[3]);
    int NZ = std::atoi(argv[4]);
    double dx = std::atof(argv[5]);
    double dy = std::atof(argv[6]);
    double dz = std::atof(argv[7]);
    if (NX <= 0 || NY <= 0 || NZ <= 0
        || !std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)
        || dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        std::cerr << "Error: mesh dimensions and cell widths must be positive and finite.\n";
        return 1;
    }
    
    double f_min = -1.0;
    double f_max = -1.0;
    double bg_rho_for_bc = -1.0; 
    int npad = 10;
    double alpha = 1.4;

    if (argc >= 10) {
        f_min = std::atof(argv[8]);
        f_max = std::atof(argv[9]);
        if (!std::isfinite(f_min) || !std::isfinite(f_max)
            || f_min <= 0.0 || f_max <= 0.0) {
            std::cerr << "Error: frequency bounds must be positive and finite.\n";
            return 1;
        }
    }
    if (argc >= 11) bg_rho_for_bc = std::atof(argv[10]);
    if (argc >= 12) npad = std::atoi(argv[11]);
    if (argc >= 13) alpha = std::atof(argv[12]);

    if (npad <= 0) {
        std::cerr << "Error: padding_layers must be positive.\n";
        return 1;
    }
    if (!std::isfinite(alpha) || alpha <= 1.0) {
        std::cerr << "Error: padding_alpha must be greater than 1.0.\n";
        return 1;
    }

    std::cout << "3D MT Forward Modeling Started (Left-Jacobi BiCGStab)...\n";
    std::cout << "Mesh: " << NX << "x" << NY << "x" << NZ << ", Cell: " << dx << "x" << dy << "x" << dz << " m\n";
    std::cout << "Padding: layers=" << npad << ", alpha=" << alpha << "\n";

    int NE = checkedMeshProduct(NX, NY, NZ, "physical mesh");
    std::vector<double> rho_vec(NE, 100.0);

    FILE *rhoFile = fopen(rhofilename.c_str(), "rb");
    if (rhoFile) {
        std::vector<float> temp_rho(NE);
        size_t read_elements = fread(temp_rho.data(), sizeof(float), NE, rhoFile);
        if (read_elements == NE) {
            for(int i=0; i<NE; ++i) rho_vec[i] = temp_rho[i];
            std::cout << "Loaded 3D rho model successfully.\n";
        } else {
            fclose(rhoFile);
            std::cerr << "Error: resistivity file contains " << read_elements
                      << " values, expected " << NE << ".\n";
            return 1;
        }
        fclose(rhoFile);
    } else {
        std::cerr << "Warning: Could not open " << rhofilename << ", using uniform 100 Ohm-m.\n";
        if (bg_rho_for_bc < 0.0) bg_rho_for_bc = 100.0;
    }

    if (bg_rho_for_bc < 0.0) {
        double sum_rho = 0.0;
        int count_boundary = 0;
        for (int k = 0; k < NZ; ++k) {
            for (int j = 0; j < NY; ++j) {
                for (int i = 0; i < NX; ++i) {
                    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1) {
                        int idx = i + j * NX + k * NX * NY;
                        sum_rho += rho_vec[idx];
                        count_boundary++;
                    }
                }
            }
        }
        bg_rho_for_bc = sum_rho / count_boundary;
        std::cout << "-> Auto-calculated Background Rho: " << bg_rho_for_bc << " Ohm-m\n";
    }
    if (!std::isfinite(bg_rho_for_bc) || bg_rho_for_bc <= 0.0
        || std::any_of(rho_vec.begin(), rho_vec.end(), [](double rho) {
               return !std::isfinite(rho) || rho <= 0.0;
           })) {
        std::cerr << "Error: resistivity values and background resistivity must be positive and finite.\n";
        return 1;
    }

    // ========================================================
    // 💡 工业级黑魔法：在内存中自动构建外延 Padding
    // ========================================================
    long long nx_pad_wide = static_cast<long long>(NX) + 2LL * npad;
    long long ny_pad_wide = static_cast<long long>(NY) + 2LL * npad;
    long long nz_pad_wide = static_cast<long long>(NZ) + npad;
    if (nx_pad_wide > INT_MAX || ny_pad_wide > INT_MAX || nz_pad_wide > INT_MAX) {
        throw std::overflow_error("padded mesh dimensions exceed 32-bit capacity");
    }
    int NX_pad = static_cast<int>(nx_pad_wide);
    int NY_pad = static_cast<int>(ny_pad_wide);
    int NZ_pad = static_cast<int>(nz_pad_wide); // No air layer; pad only below z.
    
    std::vector<double> hx(NX_pad), hy(NY_pad), hz(NZ_pad);
    for(int i=0; i<npad; i++) hx[npad - 1 - i] = dx * std::pow(alpha, i+1); // 左侧
    for(int i=0; i<NX; i++) hx[npad + i] = dx;                              // 核心区
    for(int i=0; i<npad; i++) hx[npad + NX + i] = dx * std::pow(alpha, i+1);// 右侧

    for(int i=0; i<npad; i++) hy[npad - 1 - i] = dy * std::pow(alpha, i+1); 
    for(int i=0; i<NY; i++) hy[npad + i] = dy;                              
    for(int i=0; i<npad; i++) hy[npad + NY + i] = dy * std::pow(alpha, i+1);

    for(int i=0; i<NZ; i++) hz[i] = dz;                                     // 核心区深度
    for(int i=0; i<npad; i++) hz[NZ + i] = dz * std::pow(alpha, i+1);       // 底部深层扩展

    auto invalid_width = [](double value) {
        return !std::isfinite(value) || value <= 0.0;
    };
    if (std::any_of(hx.begin(), hx.end(), invalid_width)
        || std::any_of(hy.begin(), hy.end(), invalid_width)
        || std::any_of(hz.begin(), hz.end(), invalid_width)) {
        throw std::overflow_error("padded cell widths must remain positive and finite");
    }

    // The analytical outer boundary assumes a homogeneous background. Fill the
    // padding with that same value and copy only the physical model into it.
    int NE_pad = checkedMeshProduct(NX_pad, NY_pad, NZ_pad, "padded mesh");
    long long edge_count =
        static_cast<long long>(NX_pad) * (NY_pad + 1LL) * (NZ_pad + 1LL)
        + static_cast<long long>(NX_pad + 1LL) * NY_pad * (NZ_pad + 1LL)
        + static_cast<long long>(NX_pad + 1LL) * (NY_pad + 1LL) * NZ_pad;
    if (edge_count > INT_MAX || NE_pad > INT_MAX / 144) {
        throw std::overflow_error("padded mesh exceeds CSR/topology capacity");
    }
    std::vector<double> rho_pad(NE_pad, bg_rho_for_bc);
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                rho_pad[(i + npad) + (j + npad) * NX_pad + k * NX_pad * NY_pad]
                    = rho_vec[i + j * NX + k * NX * NY];
            }
        }
    }

    std::cout << "-> Auto-Padding Applied: Mesh expanded to " << NX_pad << "x" << NY_pad << "x" << NZ_pad << std::endl;
    HostMesh3D mesh = generateMesh3D(NX_pad, NY_pad, NZ_pad, hx, hy, hz);
    int NDoF = mesh.NEdges;

    if (f_min <= 0.0 || f_max <= 0.0) {
        double skin_depth_min = dz * 2.0;                
        double total_depth = NZ * dz;
        double skin_depth_max = total_depth / 1.5;         

        f_max = bg_rho_for_bc * std::pow(503.0 / skin_depth_min, 2.0);
        f_min = bg_rho_for_bc * std::pow(503.0 / skin_depth_max, 2.0);
        if (!std::isfinite(f_min) || f_min <= 0.0
            || !std::isfinite(f_max) || f_max <= 0.0) {
            throw std::overflow_error(
                "automatic frequency bounds are outside the finite double range"
            );
        }
        if (f_min > f_max) std::swap(f_min, f_max);

        std::cout << "-> Auto-calculated Frequency Range:\n";
        std::cout << "   f_max: " << f_max << " Hz, f_min: " << f_min << " Hz\n";
    } else {
        std::cout << "-> User-provided Frequency Range:\n";
        if (f_min > f_max) std::swap(f_min, f_max); 
    }

    std::vector<double> freqs;
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
        if (f_min != f_max) freqs.push_back(f_min);
    }

    if (std::any_of(freqs.begin(), freqs.end(), [](double frequency) {
            return !std::isfinite(frequency) || frequency <= 0.0;
        })) {
        throw std::runtime_error("frequency generation produced an invalid value");
    }
    
    int n_freqs = (int)freqs.size();
    std::cout << "   Range: [" << freqs.front() << " Hz  ...  " << freqs.back() << " Hz], Points = " << n_freqs << "\n";

    std::ofstream fout_res("apparent_res_3d.txt");
    std::ofstream fout_phs("phase_3d.txt");
    std::ofstream fout_timing("solver_timing_3d.txt");
    if (!fout_res || !fout_phs || !fout_timing) {
        throw std::runtime_error("failed to open MT output files in the current directory");
    }
    fout_timing << "frequency_hz,one_time_assembly_s,combine_bc_s,solve_pol1_s,"
                   "solve_pol2_s,response_copy_s,pol1_iter,pol1_residual,"
                   "pol2_iter,pol2_residual\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> h_csrColInd, h_csrRowPtr;
    femBuild3D_CSRStructure(mesh, h_csrColInd, h_csrRowPtr);
    if (h_csrColInd.size() > static_cast<size_t>(INT_MAX)) {
        throw std::overflow_error("MT CSR matrix exceeds 32-bit index capacity");
    }
    int nnz = static_cast<int>(h_csrColInd.size());
    int surface_count = checkedMeshProduct(NX, NY, 1, "surface response grid");
    cudaStream_t stream = nullptr;

    DeviceBuffer<double> d_hx(mesh.NX);
    DeviceBuffer<double> d_hy(mesh.NY);
    DeviceBuffer<double> d_hz(mesh.NZ);
    DeviceBuffer<double> d_rho(mesh.NE);
    DeviceBuffer<double> d_stiffness(nnz);
    DeviceBuffer<double> d_conductivity_mass(nnz);
    DeviceBuffer<double> d_app_res(static_cast<size_t>(surface_count) * 2);
    DeviceBuffer<double> d_phase(static_cast<size_t>(surface_count) * 2);
    DeviceBuffer<int> d_me_edges(mesh.ME_Edges.size());
    DeviceBuffer<int> d_map_invalid(1);
    DeviceBuffer<cuDoubleComplex> d_x_pol1(NDoF);
    DeviceBuffer<cuDoubleComplex> d_x_pol2(NDoF);
    DeviceBuffer<cuDoubleComplex> d_b_pol2(NDoF);
    DeviceBuffer<cuDoubleComplex> d_fields_pol1(static_cast<size_t>(surface_count) * 4);
    DeviceBuffer<cuDoubleComplex> d_fields_pol2(static_cast<size_t>(surface_count) * 4);

    checkCuda(cudaMemcpyAsync(
        d_hx.get(), mesh.hx.data(), mesh.NX * sizeof(double),
        cudaMemcpyHostToDevice, stream
    ), "copying x cell widths to CUDA");
    checkCuda(cudaMemcpyAsync(
        d_hy.get(), mesh.hy.data(), mesh.NY * sizeof(double),
        cudaMemcpyHostToDevice, stream
    ), "copying y cell widths to CUDA");
    checkCuda(cudaMemcpyAsync(
        d_hz.get(), mesh.hz.data(), mesh.NZ * sizeof(double),
        cudaMemcpyHostToDevice, stream
    ), "copying z cell widths to CUDA");
    checkCuda(cudaMemcpyAsync(
        d_rho.get(), rho_pad.data(), mesh.NE * sizeof(double),
        cudaMemcpyHostToDevice, stream
    ), "copying resistivity to CUDA");
    checkCuda(cudaMemcpyAsync(
        d_me_edges.get(), mesh.ME_Edges.data(),
        mesh.ME_Edges.size() * sizeof(int), cudaMemcpyHostToDevice, stream
    ), "copying element topology to CUDA");
    checkCuda(cudaMemsetAsync(
        d_x_pol1.get(), 0, NDoF * sizeof(cuDoubleComplex), stream
    ), "initializing polarization 1 warm start");
    checkCuda(cudaMemsetAsync(
        d_x_pol2.get(), 0, NDoF * sizeof(cuDoubleComplex), stream
    ), "initializing polarization 2 warm start");

    std::unique_ptr<GpuBiCGStabContext, SolverContextDeleter> solver_context(
        create_gpu_bicgstab_context(NDoF, nnz)
    );
    set_gpu_bicgstab_stream(solver_context.get(), stream);
    initialize_gpu_bicgstab_structure(
        solver_context.get(), h_csrRowPtr, h_csrColInd
    );
    const int* d_csr_row_ptr = get_gpu_bicgstab_row_ptr_device(solver_context.get());
    const int* d_csr_col_ind = get_gpu_bicgstab_col_ind_device(solver_context.get());
    cuDoubleComplex* d_csr_values =
        get_gpu_bicgstab_matrix_values_device(solver_context.get());
    cuDoubleComplex* d_b_pol1 = get_gpu_bicgstab_rhs_device(solver_context.get());
    std::unique_ptr<BoundaryConditionPlan, BoundaryPlanDeleter> boundary_plan(
        createBoundaryConditionPlan3D(mesh, h_csrColInd, h_csrRowPtr)
    );

    auto assembly_start = std::chrono::high_resolution_clock::now();
    gpu_assemble_component_matrices(
        mesh.NE, mesh.NX, mesh.NY, nnz,
        d_hx.get(), d_hy.get(), d_hz.get(), d_rho.get(),
        d_me_edges.get(), d_csr_row_ptr, d_csr_col_ind,
        d_stiffness.get(), d_conductivity_mass.get(),
        d_map_invalid.get(), stream
    );
    checkLastCudaError("assembling frequency-independent MT matrices");
    int map_invalid_host = 0;
    checkCuda(cudaMemcpyAsync(
        &map_invalid_host, d_map_invalid.get(), sizeof(int),
        cudaMemcpyDeviceToHost, stream
    ), "reading CSR assembly validation flag");
    checkCuda(cudaStreamSynchronize(stream), "synchronizing MT component assembly");
    auto assembly_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_elapsed = assembly_end - assembly_start;
    if (map_invalid_host != 0) {
        throw std::runtime_error(
            "MT assembly found a missing CSR entry; mesh topology and CSR disagree"
        );
    }
    std::cout << "-> Frequency-independent matrix assembly: "
              << assembly_elapsed.count() << " s\n";

    std::vector<double> app_res_slice(static_cast<size_t>(surface_count) * 2);
    std::vector<double> phase_slice(static_cast<size_t>(surface_count) * 2);
    bool all_converged = true;

    for (int ff = 0; ff < n_freqs; ff++) {
        double freq = freqs[ff];
        std::cout << "\n=== Processing Frequency " << ff + 1 << "/" << n_freqs << ": " << freq << " Hz ===" << std::endl;

        auto combine_start = std::chrono::high_resolution_clock::now();
        gpu_combine_component_matrices(
            nnz, freq, d_stiffness.get(), d_conductivity_mass.get(),
            d_csr_values, stream
        );
        applyBoundaryConditions3DDeviceDual(
            freq, bg_rho_for_bc, d_csr_values, d_b_pol1, d_b_pol2.get(),
            boundary_plan.get(), stream
        );
        checkLastCudaError("combining MT matrix and applying boundary conditions");
        checkCuda(cudaStreamSynchronize(stream), "synchronizing MT boundary conditions");
        auto combine_end = std::chrono::high_resolution_clock::now();

        std::cout << "  -> Solving Polarization 1 (Ex) using BiCGStab..." << std::endl;
        auto solve_start_pol1 = std::chrono::high_resolution_clock::now();
        SolverStats pol1Stats;
        bool pol1Converged = gpu_complex_solver_bicgstab_device(
            solver_context.get(), d_b_pol1, d_x_pol1.get(),
            true, 1e-7, 50000, &pol1Stats
        );
        checkCuda(cudaStreamSynchronize(stream), "synchronizing polarization 1 solve");
        auto solve_end_pol1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> solve_elapsed_pol1 = solve_end_pol1 - solve_start_pol1;
        std::cout << "     Polarization 1 stats: iterations=" << pol1Stats.iterations
                  << ", residual=" << pol1Stats.finalResidual
                  << ", elapsed=" << solve_elapsed_pol1.count() << " s" << std::endl;
        if (!pol1Converged || !std::isfinite(pol1Stats.finalResidual)) {
            std::cerr << "Polarization 1 failed at " << freq << " Hz.\n";
            all_converged = false;
            break;
        }

        std::cout << "  -> Solving Polarization 2 (Ey) using BiCGStab..." << std::endl;
        auto solve_start_pol2 = std::chrono::high_resolution_clock::now();
        SolverStats pol2Stats;
        bool pol2Converged = gpu_complex_solver_bicgstab_device(
            solver_context.get(), d_b_pol2.get(), d_x_pol2.get(),
            false, 1e-7, 50000, &pol2Stats
        );
        checkCuda(cudaStreamSynchronize(stream), "synchronizing polarization 2 solve");
        auto solve_end_pol2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> solve_elapsed_pol2 = solve_end_pol2 - solve_start_pol2;
        std::cout << "     Polarization 2 stats: iterations=" << pol2Stats.iterations
                  << ", residual=" << pol2Stats.finalResidual
                  << ", elapsed=" << solve_elapsed_pol2.count() << " s" << std::endl;
        if (!pol2Converged || !std::isfinite(pol2Stats.finalResidual)) {
            std::cerr << "Polarization 2 failed at " << freq << " Hz.\n";
            all_converged = false;
            break;
        }

        auto response_start = std::chrono::high_resolution_clock::now();
        gpu_extract_surface_fields_2d(
            mesh.NX, mesh.NY, NX, NY, npad, 0, freq,
            d_hx.get(), d_hy.get(), d_hz.get(), d_me_edges.get(),
            d_x_pol1.get(), d_fields_pol1.get(), stream
        );
        gpu_extract_surface_fields_2d(
            mesh.NX, mesh.NY, NX, NY, npad, 0, freq,
            d_hx.get(), d_hy.get(), d_hz.get(), d_me_edges.get(),
            d_x_pol2.get(), d_fields_pol2.get(), stream
        );
        gpu_compute_mt_responses(
            NX, NY, freq, d_fields_pol1.get(), d_fields_pol2.get(),
            d_app_res.get(), d_phase.get(), stream
        );
        checkLastCudaError("extracting MT surface responses");
        checkCuda(cudaMemcpyAsync(
            app_res_slice.data(), d_app_res.get(),
            app_res_slice.size() * sizeof(double), cudaMemcpyDeviceToHost, stream
        ), "copying apparent resistivity from CUDA");
        checkCuda(cudaMemcpyAsync(
            phase_slice.data(), d_phase.get(),
            phase_slice.size() * sizeof(double), cudaMemcpyDeviceToHost, stream
        ), "copying phase from CUDA");
        checkCuda(cudaStreamSynchronize(stream), "synchronizing MT response copy");
        auto response_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> combine_elapsed = combine_end - combine_start;
        std::chrono::duration<double> response_elapsed = response_end - response_start;
        fout_timing << freq << ","
                    << (ff == 0 ? assembly_elapsed.count() : 0.0) << ","
                    << combine_elapsed.count() << ","
                    << solve_elapsed_pol1.count() << ","
                    << solve_elapsed_pol2.count() << ","
                    << response_elapsed.count() << ","
                    << pol1Stats.iterations << ","
                    << pol1Stats.finalResidual << ","
                    << pol2Stats.iterations << ","
                    << pol2Stats.finalResidual << "\n";

        for (int i = 0; i < surface_count; ++i) {
            int x_idx = i % NX;
            int y_idx = i / NX;
            int tensor_idx = x_idx * NY + y_idx;
            fout_res << freq << " " << i << " "
                     << app_res_slice[2 * tensor_idx]
                     << " " << app_res_slice[2 * tensor_idx + 1]
                     << "\n";
            fout_phs << freq << " " << i << " "
                     << phase_slice[2 * tensor_idx]
                     << " " << phase_slice[2 * tensor_idx + 1]
                     << "\n";
        }
    }

    fout_res.close();
    fout_phs.close();
    fout_timing.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nAll done! Total time: " << elapsed.count() << " s\n";
    return all_converged ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
