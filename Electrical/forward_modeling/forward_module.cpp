#include <torch/extension.h>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <ATen/TensorIndexing.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

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
    cudaStream_t stream);

extern "C" void gpu_combine_component_matrices(
    int nnz,
    double freq,
    const double* d_stiffness,
    const double* d_conductivityMass,
    cuDoubleComplex* d_values,
    cudaStream_t stream);

extern "C" void gpu_extract_surface_fields_2d(
    int nxPad, int nyPad, int coreNx, int coreNy, int npad,
    int surfaceLayer, double freq,
    const double* d_hx, const double* d_hy, const double* d_hz,
    const int* d_meEdges, const cuDoubleComplex* d_x, cuDoubleComplex* d_fields,
    cudaStream_t stream);

extern "C" void gpu_compute_mt_responses(
    int nx, int ny, double freq,
    const cuDoubleComplex* d_fields_pol1,
    const cuDoubleComplex* d_fields_pol2,
    double* d_app_res_slice,
    double* d_phase_slice,
    cudaStream_t stream);

namespace {

double estimate_boundary_background_rho(const torch::Tensor& rho_tensor) {
    const auto nx = rho_tensor.size(0);
    const auto ny = rho_tensor.size(1);
    const auto nz = rho_tensor.size(2);

    auto boundary_sum = rho_tensor.sum();
    int64_t boundary_count = nx * ny * nz;
    if (nx > 2 && ny > 2 && nz > 2) {
        boundary_sum = boundary_sum - rho_tensor.slice(0, 1, nx - 1)
            .slice(1, 1, ny - 1)
            .slice(2, 1, nz - 1)
            .sum();
        boundary_count -= (nx - 2) * (ny - 2) * (nz - 2);
    }
    return (boundary_sum / static_cast<double>(boundary_count)).item<double>();
}

struct BoundaryPlanDeleter {
    void operator()(BoundaryConditionPlan* plan) const {
        destroyBoundaryConditionPlan3D(plan);
    }
};

struct SolverContextDeleter {
    void operator()(GpuBiCGStabContext* context) const {
        destroy_gpu_bicgstab_context(context);
    }
};

struct GeometryCacheKey {
    int nx;
    int ny;
    int nz;
    int npad;
    int air_layers;
    double dx;
    double dy;
    double dz;
    double padding_growth;

    bool operator==(const GeometryCacheKey& other) const {
        return nx == other.nx && ny == other.ny && nz == other.nz
            && npad == other.npad && air_layers == other.air_layers
            && dx == other.dx && dy == other.dy
            && dz == other.dz && padding_growth == other.padding_growth;
    }
};

struct CachedHostGeometry {
    HostMesh3D mesh;
    std::vector<int> csr_col_ind;
    std::vector<int> csr_row_ptr;
};

struct HostGeometryCacheState {
    std::mutex mutex;
    std::unique_ptr<GeometryCacheKey> key;
    std::shared_ptr<const CachedHostGeometry> geometry;
    std::uint64_t hits = 0;
    std::uint64_t builds = 0;
};

constexpr std::size_t max_cached_host_geometry_bytes = 256ULL * 1024ULL * 1024ULL;

std::size_t retained_host_geometry_bytes(const CachedHostGeometry& geometry) {
    return geometry.mesh.hx.capacity() * sizeof(double)
        + geometry.mesh.hy.capacity() * sizeof(double)
        + geometry.mesh.hz.capacity() * sizeof(double)
        + geometry.mesh.ME_Edges.capacity() * sizeof(int)
        + geometry.csr_col_ind.capacity() * sizeof(int)
        + geometry.csr_row_ptr.capacity() * sizeof(int);
}

HostGeometryCacheState& host_geometry_cache_state() {
    static HostGeometryCacheState state;
    return state;
}

std::shared_ptr<const CachedHostGeometry> get_cached_host_geometry(
    int nx,
    int ny,
    int nz,
    int npad,
    int air_layers,
    double dx,
    double dy,
    double dz,
    double padding_growth)
{
    const GeometryCacheKey requested{
        nx, ny, nz, npad, air_layers, dx, dy, dz, padding_growth
    };
    auto& cache = host_geometry_cache_state();

    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.key && *cache.key == requested) {
            ++cache.hits;
            return cache.geometry;
        }
        // Release an inactive entry before constructing a replacement. Active
        // callers retain their own shared_ptr and therefore remain valid.
        cache.key.reset();
        cache.geometry.reset();
    }

    const int nx_pad = nx + 2 * npad;
    const int ny_pad = ny + 2 * npad;
    const int nz_pad = air_layers + nz + npad;
    std::vector<double> hx(nx_pad), hy(ny_pad), hz(nz_pad);
    for (int i = 0; i < npad; ++i) {
        hx[npad - 1 - i] = dx * std::pow(padding_growth, i + 1);
        hx[npad + nx + i] = dx * std::pow(padding_growth, i + 1);
        hy[npad - 1 - i] = dy * std::pow(padding_growth, i + 1);
        hy[npad + ny + i] = dy * std::pow(padding_growth, i + 1);
        hz[air_layers + nz + i] = dz * std::pow(padding_growth, i + 1);
    }
    std::fill(hx.begin() + npad, hx.begin() + npad + nx, dx);
    std::fill(hy.begin() + npad, hy.begin() + npad + ny, dy);
    std::fill(hz.begin(), hz.begin() + air_layers + nz, dz);

    auto built = std::make_shared<CachedHostGeometry>();
    built->mesh = generateMesh3D(nx_pad, ny_pad, nz_pad, hx, hy, hz);
    femBuild3D_CSRStructure(
        built->mesh, built->csr_col_ind, built->csr_row_ptr
    );
    // Node connectivity is not consumed by this edge-element extension. Drop it
    // from the retained entry so a topology cache does not keep dead memory.
    std::vector<int>().swap(built->mesh.ME_Nodes);
    const bool cacheable = retained_host_geometry_bytes(*built)
        <= max_cached_host_geometry_bytes;

    std::lock_guard<std::mutex> lock(cache.mutex);
    ++cache.builds;
    // Another thread may have completed the same build while this one ran.
    if (cache.key && *cache.key == requested) {
        ++cache.hits;
        return cache.geometry;
    }
    if (cacheable) {
        cache.key = std::make_unique<GeometryCacheKey>(requested);
        cache.geometry = built;
    }
    return built;
}

std::pair<std::uint64_t, std::uint64_t> host_geometry_cache_stats() {
    auto& cache = host_geometry_cache_state();
    std::lock_guard<std::mutex> lock(cache.mutex);
    return {cache.hits, cache.builds};
}

void clear_host_geometry_cache() {
    auto& cache = host_geometry_cache_state();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.key.reset();
    cache.geometry.reset();
    cache.hits = 0;
    cache.builds = 0;
}

void checkCuda(cudaError_t status, const char* operation) {
    TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

void checkLastCudaError(const char* operation) {
    checkCuda(cudaGetLastError(), operation);
}

torch::Tensor build_padded_rho_tensor(
    const torch::Tensor& rho_tensor,
    int npad,
    double background_rho,
    int air_layers,
    double air_resistivity)
{
    using namespace torch::indexing;
    const auto nx = rho_tensor.size(0);
    const auto ny = rho_tensor.size(1);
    const auto nz = rho_tensor.size(2);

    auto padded = torch::full(
        {nx + 2 * npad, ny + 2 * npad, air_layers + nz + npad},
        background_rho,
        rho_tensor.options()
    );
    if (air_layers > 0) {
        padded.index_put_(
            {Slice(), Slice(), Slice(0, air_layers)},
            air_resistivity
        );
    }
    padded.index_put_(
        {
            Slice(npad, npad + nx),
            Slice(npad, npad + ny),
            Slice(air_layers, air_layers + nz),
        },
        rho_tensor
    );
    return padded.contiguous();
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> compute_mt_3d(
    torch::Tensor rho_tensor,
    double dx,
    double dy,
    double dz,
    std::vector<double> freqs,
    double supplied_background_rho = std::numeric_limits<double>::quiet_NaN(),
    int air_layers = 0,
    double air_resistivity = 1.0e8,
    double solver_tolerance = 1.0e-7,
    int solver_max_iterations = 50000,
    bool frequency_continuation = true,
    int padding_layers = 10,
    double padding_growth = 1.4,
    bool rho_is_validated = false)
{
    TORCH_CHECK(rho_tensor.dim() == 3, "rho_tensor must be 3D with shape (NX, NY, NZ)");
    TORCH_CHECK(
        !rho_tensor.requires_grad(),
        "MT forward modeling is forward-only and does not implement autograd"
    );
    TORCH_CHECK(
        rho_tensor.size(0) > 0 && rho_tensor.size(1) > 0 && rho_tensor.size(2) > 0,
        "rho_tensor dimensions NX, NY, and NZ must all be positive"
    );
    TORCH_CHECK(
        !rho_tensor.is_complex() && rho_tensor.scalar_type() != torch::kBool,
        "rho_tensor must contain real-valued resistivity data"
    );
    TORCH_CHECK(std::isfinite(dx) && dx > 0.0, "dx must be positive and finite");
    TORCH_CHECK(std::isfinite(dy) && dy > 0.0, "dy must be positive and finite");
    TORCH_CHECK(std::isfinite(dz) && dz > 0.0, "dz must be positive and finite");
    TORCH_CHECK(air_layers >= 0, "air_layers must be nonnegative");
    TORCH_CHECK(
        std::isfinite(air_resistivity) && air_resistivity > 0.0,
        "air_resistivity must be positive and finite"
    );
    TORCH_CHECK(
        std::isfinite(solver_tolerance) && solver_tolerance > 0.0,
        "solver_tolerance must be positive and finite"
    );
    TORCH_CHECK(
        solver_max_iterations > 0,
        "solver_max_iterations must be positive"
    );
    TORCH_CHECK(padding_layers > 0, "padding_layers must be positive");
    TORCH_CHECK(
        std::isfinite(padding_growth) && padding_growth > 1.0,
        "padding_growth must be finite and greater than 1.0"
    );
    for (size_t index = 0; index < freqs.size(); ++index) {
        TORCH_CHECK(
            std::isfinite(freqs[index]) && freqs[index] > 0.0,
            "freqs[", index, "] must be positive and finite"
        );
    }
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    TORCH_CHECK(
        cuda_status == cudaSuccess && device_count > 0,
        "mt_forward_cuda requires a working CUDA runtime and at least one visible GPU. "
        "cudaGetDeviceCount failed with: ", cudaGetErrorString(cuda_status)
    );

    const bool return_on_cuda = rho_tensor.is_cuda();
    const c10::DeviceIndex device_index = return_on_cuda
        ? rho_tensor.get_device()
        : c10::cuda::current_device();
    c10::cuda::CUDAGuard device_guard(device_index);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

    auto rho_work = rho_tensor.to(
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, device_index)
    ).contiguous();
    if (!rho_is_validated) {
        auto invalid_rho = torch::logical_or(
            torch::logical_not(torch::isfinite(rho_work)),
            rho_work <= 0.0
        );
        TORCH_CHECK(
            !invalid_rho.any().item<bool>(),
            "rho_tensor resistivity values must all be positive and finite"
        );
    }
    
    const int npad = padding_layers;
    constexpr int maxInt = std::numeric_limits<int>::max();
    TORCH_CHECK(
        rho_work.size(0) + 2LL * npad <= maxInt
            && rho_work.size(1) + 2LL * npad <= maxInt
            && rho_work.size(2) + static_cast<int64_t>(npad) + air_layers <= maxInt,
        "rho_tensor dimensions are too large for the solver's 32-bit mesh indices"
    );
    int NX = static_cast<int>(rho_work.size(0));
    int NY = static_cast<int>(rho_work.size(1));
    int NZ = static_cast<int>(rho_work.size(2));

    double bg_rho_for_bc = std::isnan(supplied_background_rho)
        ? estimate_boundary_background_rho(rho_work)
        : supplied_background_rho;
    TORCH_CHECK(
        std::isfinite(bg_rho_for_bc) && bg_rho_for_bc > 0.0,
        "background_rho must be positive and finite"
    );

    const double alpha = padding_growth;
    TORCH_CHECK(
        std::isfinite(dx * std::pow(alpha, npad))
            && std::isfinite(dy * std::pow(alpha, npad))
            && std::isfinite(dz * std::pow(alpha, npad)),
        "padding cell widths overflow; reduce padding_layers or padding_growth"
    );
    
    int NX_pad = NX + 2 * npad;
    int NY_pad = NY + 2 * npad;
    int NZ_pad = air_layers + NZ + npad;
    int64_t element_count = static_cast<int64_t>(NX_pad) * NY_pad * NZ_pad;
    int64_t edge_count =
        static_cast<int64_t>(NX_pad)
            * (static_cast<int64_t>(NY_pad) + 1)
            * (static_cast<int64_t>(NZ_pad) + 1)
        + (static_cast<int64_t>(NX_pad) + 1)
            * NY_pad
            * (static_cast<int64_t>(NZ_pad) + 1)
        + (static_cast<int64_t>(NX_pad) + 1)
            * (static_cast<int64_t>(NY_pad) + 1)
            * NZ_pad;
    TORCH_CHECK(
        element_count <= maxInt && edge_count <= maxInt
            && element_count <= maxInt / 144,
        "padded MT mesh exceeds the solver's 32-bit CSR/index capacity"
    );
    
    auto rho_pad_tensor = build_padded_rho_tensor(
        rho_work, npad, bg_rho_for_bc, air_layers, air_resistivity
    );
    // The FEM/CUDA kernels expect the flattened resistivity ordering used by the
    // original MTForward3D code: idx = x + y * NX + z * NX * NY (x-fastest).
    // A contiguous torch tensor with logical shape (NX, NY, NZ) is laid out with
    // z-fastest storage, so we materialize a (NZ, NY, NX)-contiguous view before
    // exposing its raw storage pointer to the solver.
    auto rho_pad_linear_tensor = rho_pad_tensor.permute({2, 1, 0}).contiguous();

    auto geometry = get_cached_host_geometry(
        NX, NY, NZ, npad, air_layers, dx, dy, dz, alpha
    );
    const HostMesh3D& mesh = geometry->mesh;
    int NDoF = mesh.NEdges;
    
    if (freqs.empty()) {
        double skin_depth_min = dz * 2.0;
        double total_depth = NZ * dz;
        double skin_depth_max = total_depth / 1.5;
        double f_max = bg_rho_for_bc * std::pow(503.0 / skin_depth_min, 2.0);
        double f_min = bg_rho_for_bc * std::pow(503.0 / skin_depth_max, 2.0);
        TORCH_CHECK(
            std::isfinite(f_min) && f_min > 0.0
                && std::isfinite(f_max) && f_max > 0.0,
            "automatic MT frequency bounds are outside the finite double range; "
            "rescale the cell widths or provide explicit frequencies"
        );
        if (f_min > f_max) {
            std::swap(f_min, f_max);
        }

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
            if (f_min != f_max) {
                freqs.push_back(f_min);
            }
        }
    }

    TORCH_CHECK(
        freqs.size() <= static_cast<size_t>(maxInt),
        "too many MT frequencies for 32-bit output indexing"
    );
    int n_freqs = static_cast<int>(freqs.size());
    std::vector<int> frequency_solve_order(n_freqs);
    for (int index = 0; index < n_freqs; ++index) {
        frequency_solve_order[index] = index;
    }
    // When enabled, continuation from high to low frequency can give the
    // difficult low-frequency systems a better initial guess. Results are still
    // written at the original indices, so sorting does not change the API.
    std::stable_sort(
        frequency_solve_order.begin(),
        frequency_solve_order.end(),
        [&](int left, int right) { return freqs[left] > freqs[right]; }
    );

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(rho_work.device());
    // Keep GMESDataset's repository-wide convention: [n_freq, NX, NY, 2].
    torch::Tensor app_res = torch::empty({n_freqs, NX, NY, 2}, options);
    torch::Tensor phase = torch::empty({n_freqs, NX, NY, 2}, options);

    std::cout << "3D MT Forward Modeling Started (Left-Jacobi BiCGStab)...\n";
    std::cout << "Mesh: " << NX << "x" << NY << "x" << NZ << ", Cell: " << dx << "x" << dy << "x" << dz << " m\n";
    std::cout << "-> Boundary Background Rho: " << bg_rho_for_bc << " Ohm-m\n";
    std::cout << "-> Padding Applied: Mesh expanded to " << NX_pad << "x" << NY_pad << "x" << NZ_pad << std::endl;
    if (air_layers > 0) {
        std::cout << "-> Explicit air: " << air_layers << " layers, rho="
                  << air_resistivity << " Ohm-m" << std::endl;
    }
    std::cout << "   Range: [" << freqs.front() << " Hz  ...  " << freqs.back() << " Hz], Points = " << n_freqs << "\n";

    const auto& h_csrColInd = geometry->csr_col_ind;
    const auto& h_csrRowPtr = geometry->csr_row_ptr;
    TORCH_CHECK(
        h_csrColInd.size() <= static_cast<size_t>(maxInt),
        "MT CSR matrix exceeds 32-bit index capacity"
    );
    int nnz = static_cast<int>(h_csrColInd.size());

    auto int_options = options.dtype(torch::kInt32);
    auto complex_options = options.dtype(torch::kComplexDouble);
    auto d_hx_tensor = torch::empty({mesh.NX}, options);
    auto d_hy_tensor = torch::empty({mesh.NY}, options);
    auto d_hz_tensor = torch::empty({mesh.NZ}, options);
    auto d_me_edges_tensor = torch::empty(
        {static_cast<int64_t>(mesh.ME_Edges.size())}, int_options
    );
    auto d_map_invalid_tensor = torch::zeros({1}, int_options);
    auto d_stiffness_tensor = torch::empty({nnz}, options);
    auto d_conductivity_mass_tensor = torch::empty({nnz}, options);
    auto d_x_pol1_tensor = torch::zeros({NDoF}, complex_options);
    auto d_x_pol2_tensor = torch::zeros({NDoF}, complex_options);
    auto d_b_pol2_tensor = torch::empty({NDoF}, complex_options);
    auto d_fields_pol1_tensor = torch::empty(
        {static_cast<int64_t>(NX) * NY * 4}, complex_options
    );
    auto d_fields_pol2_tensor = torch::empty(
        {static_cast<int64_t>(NX) * NY * 4}, complex_options
    );

    double* d_hx = d_hx_tensor.data_ptr<double>();
    double* d_hy = d_hy_tensor.data_ptr<double>();
    double* d_hz = d_hz_tensor.data_ptr<double>();
    int* d_me_edges = d_me_edges_tensor.data_ptr<int>();
    int* d_map_invalid = d_map_invalid_tensor.data_ptr<int>();
    double* d_stiffness = d_stiffness_tensor.data_ptr<double>();
    double* d_conductivity_mass = d_conductivity_mass_tensor.data_ptr<double>();
    const double* d_rho = rho_pad_linear_tensor.data_ptr<double>();
    auto* d_x_pol1 = reinterpret_cast<cuDoubleComplex*>(d_x_pol1_tensor.data_ptr());
    auto* d_x_pol2 = reinterpret_cast<cuDoubleComplex*>(d_x_pol2_tensor.data_ptr());
    auto* d_b_pol2 = reinterpret_cast<cuDoubleComplex*>(d_b_pol2_tensor.data_ptr());
    auto* d_fields_pol1 = reinterpret_cast<cuDoubleComplex*>(d_fields_pol1_tensor.data_ptr());
    auto* d_fields_pol2 = reinterpret_cast<cuDoubleComplex*>(d_fields_pol2_tensor.data_ptr());
    double* d_app_res = app_res.data_ptr<double>();
    double* d_phase = phase.data_ptr<double>();

    checkCuda(cudaMemcpyAsync(
        d_hx, mesh.hx.data(), mesh.NX * sizeof(double), cudaMemcpyHostToDevice, stream
    ), "copying x cell widths to CUDA");
    checkCuda(cudaMemcpyAsync(
        d_hy, mesh.hy.data(), mesh.NY * sizeof(double), cudaMemcpyHostToDevice, stream
    ), "copying y cell widths to CUDA");
    checkCuda(cudaMemcpyAsync(
        d_hz, mesh.hz.data(), mesh.NZ * sizeof(double), cudaMemcpyHostToDevice, stream
    ), "copying z cell widths to CUDA");
    checkCuda(cudaMemcpyAsync(
        d_me_edges, mesh.ME_Edges.data(), mesh.ME_Edges.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream
    ), "copying element topology to CUDA");

    std::unique_ptr<GpuBiCGStabContext, SolverContextDeleter> solver_context(
        create_gpu_bicgstab_context(NDoF, nnz)
    );
    set_gpu_bicgstab_stream(solver_context.get(), stream);
    initialize_gpu_bicgstab_structure(solver_context.get(), h_csrRowPtr, h_csrColInd);
    const int* d_csr_row_ptr = get_gpu_bicgstab_row_ptr_device(solver_context.get());
    const int* d_csr_col_ind = get_gpu_bicgstab_col_ind_device(solver_context.get());
    cuDoubleComplex* d_csr_values = get_gpu_bicgstab_matrix_values_device(solver_context.get());
    cuDoubleComplex* d_b_pol1 = get_gpu_bicgstab_rhs_device(solver_context.get());
    std::unique_ptr<BoundaryConditionPlan, BoundaryPlanDeleter> boundary_plan(
        createBoundaryConditionPlan3D(
            mesh, h_csrColInd, h_csrRowPtr, air_layers
        )
    );

    gpu_assemble_component_matrices(
        mesh.NE, mesh.NX, mesh.NY, nnz,
        d_hx, d_hy, d_hz, d_rho,
        d_me_edges, d_csr_row_ptr, d_csr_col_ind,
        d_stiffness, d_conductivity_mass, d_map_invalid, stream
    );
    checkLastCudaError("assembling frequency-independent MT matrices");
    int map_invalid_host = 0;
    checkCuda(cudaMemcpyAsync(
        &map_invalid_host, d_map_invalid, sizeof(int), cudaMemcpyDeviceToHost, stream
    ), "reading the CSR assembly validation flag");
    checkCuda(cudaStreamSynchronize(stream), "synchronizing CSR assembly validation");
    TORCH_CHECK(
        map_invalid_host == 0,
        "MT assembly found a missing CSR entry; mesh topology and CSR structure disagree"
    );

    for (int solve_index = 0; solve_index < n_freqs; ++solve_index) {
        int ff = frequency_solve_order[solve_index];
        double freq = freqs[ff];
        std::cout << "\n=== Processing Frequency " << solve_index + 1 << "/"
                  << n_freqs << ": " << freq << " Hz (output index "
                  << ff << ") ===" << std::endl;

        if (!frequency_continuation && solve_index > 0) {
            checkCuda(cudaMemsetAsync(
                d_x_pol1, 0, static_cast<size_t>(NDoF) * sizeof(cuDoubleComplex), stream
            ), "clearing polarization 1 continuation field");
            checkCuda(cudaMemsetAsync(
                d_x_pol2, 0, static_cast<size_t>(NDoF) * sizeof(cuDoubleComplex), stream
            ), "clearing polarization 2 continuation field");
        }

        gpu_combine_component_matrices(
            nnz, freq, d_stiffness, d_conductivity_mass, d_csr_values, stream
        );
        applyBoundaryConditions3DDeviceDual(
            freq, bg_rho_for_bc, d_csr_values, d_b_pol1, d_b_pol2,
            boundary_plan.get(), stream
        );
        checkLastCudaError("combining the MT matrix and applying boundary conditions");
        std::cout << "  -> Solving Polarization 1 (Ex) using BiCGStab..." << std::endl;
        SolverStats pol1Stats;
        bool pol1Converged = gpu_complex_solver_bicgstab_device(
            solver_context.get(), d_b_pol1, d_x_pol1, true,
            solver_tolerance, solver_max_iterations, &pol1Stats
        );
        TORCH_CHECK(
            pol1Converged && std::isfinite(pol1Stats.finalResidual),
            "MT BiCGStab failed for polarization 1 at ", freq,
            " Hz after ", pol1Stats.iterations,
            " iterations; residual=", pol1Stats.finalResidual
        );
        std::cout << "     Polarization 1 stats: iterations=" << pol1Stats.iterations
                  << ", residual=" << pol1Stats.finalResidual << std::endl;
        gpu_extract_surface_fields_2d(
            mesh.NX, mesh.NY, NX, NY, npad, air_layers, freq,
            d_hx, d_hy, d_hz, d_me_edges, d_x_pol1, d_fields_pol1, stream
        );

        std::cout << "  -> Solving Polarization 2 (Ey) using BiCGStab..." << std::endl;
        SolverStats pol2Stats;
        bool pol2Converged = gpu_complex_solver_bicgstab_device(
            solver_context.get(), d_b_pol2, d_x_pol2, false,
            solver_tolerance, solver_max_iterations, &pol2Stats
        );
        TORCH_CHECK(
            pol2Converged && std::isfinite(pol2Stats.finalResidual),
            "MT BiCGStab failed for polarization 2 at ", freq,
            " Hz after ", pol2Stats.iterations,
            " iterations; residual=", pol2Stats.finalResidual
        );
        std::cout << "     Polarization 2 stats: iterations=" << pol2Stats.iterations
                  << ", residual=" << pol2Stats.finalResidual << std::endl;
        gpu_extract_surface_fields_2d(
            mesh.NX, mesh.NY, NX, NY, npad, air_layers, freq,
            d_hx, d_hy, d_hz, d_me_edges, d_x_pol2, d_fields_pol2, stream
        );

        gpu_compute_mt_responses(
            NX, NY, freq,
            d_fields_pol1,
            d_fields_pol2,
            d_app_res + static_cast<size_t>(ff) * NX * NY * 2,
            d_phase + static_cast<size_t>(ff) * NX * NY * 2,
            stream
        );
        checkLastCudaError("extracting MT fields and responses");
    }

    if (!return_on_cuda) {
        return std::make_tuple(app_res.cpu(), phase.cpu());
    }
    return std::make_tuple(app_res, phase);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_host_geometry_cache_stats", &host_geometry_cache_stats);
    m.def("_clear_host_geometry_cache", &clear_host_geometry_cache);
    m.def(
        "compute_mt_3d",
        [](torch::Tensor rho_tensor,
           double dx,
           double dy,
           double dz,
           std::vector<double> freqs,
           double background_rho,
           int air_layers,
           double air_resistivity,
           double solver_tolerance,
           int solver_max_iterations,
           bool frequency_continuation,
           int padding_layers,
           double padding_growth) {
            return compute_mt_3d(
                std::move(rho_tensor), dx, dy, dz, std::move(freqs),
                background_rho, air_layers, air_resistivity,
                solver_tolerance, solver_max_iterations,
                frequency_continuation, padding_layers, padding_growth, false
            );
        },
        "MT 3D Forward Modeling",
        pybind11::arg("rho_tensor"),
        pybind11::arg("dx"),
        pybind11::arg("dy"),
        pybind11::arg("dz"),
        pybind11::arg("freqs"),
        pybind11::arg("background_rho") = std::numeric_limits<double>::quiet_NaN(),
        pybind11::arg("air_layers") = 0,
        pybind11::arg("air_resistivity") = 1.0e8,
        pybind11::arg("solver_tolerance") = 1.0e-7,
        pybind11::arg("solver_max_iterations") = 50000,
        pybind11::arg("frequency_continuation") = true,
        pybind11::arg("padding_layers") = 10,
        pybind11::arg("padding_growth") = 1.4
    );
    m.def(
        "_compute_mt_3d_validated",
        [](torch::Tensor rho_tensor,
           double dx,
           double dy,
           double dz,
           std::vector<double> freqs,
           double background_rho,
           int air_layers,
           double air_resistivity,
           double solver_tolerance,
           int solver_max_iterations,
           bool frequency_continuation,
           int padding_layers,
           double padding_growth) {
            return compute_mt_3d(
                std::move(rho_tensor), dx, dy, dz, std::move(freqs),
                background_rho, air_layers, air_resistivity,
                solver_tolerance, solver_max_iterations,
                frequency_continuation, padding_layers, padding_growth, true
            );
        },
        pybind11::arg("rho_tensor"),
        pybind11::arg("dx"),
        pybind11::arg("dy"),
        pybind11::arg("dz"),
        pybind11::arg("freqs"),
        pybind11::arg("background_rho"),
        pybind11::arg("air_layers") = 0,
        pybind11::arg("air_resistivity") = 1.0e8,
        pybind11::arg("solver_tolerance") = 1.0e-7,
        pybind11::arg("solver_max_iterations") = 50000,
        pybind11::arg("frequency_continuation") = true,
        pybind11::arg("padding_layers") = 10,
        pybind11::arg("padding_growth") = 1.4
    );
}
