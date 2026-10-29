import sys
from pathlib import Path

import pytest
import torch


GEMS_ROOT = Path(__file__).resolve().parents[3]
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from Electrical.forward_modeling import mt_forward as mt_module  # noqa: E402
from Electrical.forward_modeling.mt_forward import MTForward3D  # noqa: E402


CUDA_EXTENSION_AVAILABLE = (
    torch.cuda.is_available() and mt_module.mt_forward_cuda is not None
)


def _wrapped_phase_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.remainder(actual - expected + 180.0, 360.0) - 180.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_extension_imports_when_cuda_is_available():
    assert mt_module.mt_forward_cuda is not None, (
        "CUDA is available, but the mt_forward_cuda extension could not be imported"
    )


@pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE,
    reason="CUDA and the compiled mt_forward_cuda extension are required",
)
def test_direct_cuda_binding_rejects_autograd_input():
    rho = torch.ones((1, 1, 1), dtype=torch.float64, requires_grad=True)

    with pytest.raises(RuntimeError, match="forward-only.*autograd"):
        mt_module.mt_forward_cuda.compute_mt_3d(
            rho, 100.0, 100.0, 100.0, [4.0], 100.0
        )


@pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE,
    reason="CUDA and the compiled mt_forward_cuda extension are required",
)
def test_homogeneous_halfspace_matches_analytic_response_on_cuda():
    rho0 = 100.0
    device = torch.device("cuda", torch.cuda.current_device())
    rho = torch.full((1, 1, 1), rho0, dtype=torch.float64, device=device)
    solver = MTForward3D(freqs=[4.0], dx=100.0, dy=100.0, dz=100.0)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        app_res, phase = solver(rho)
    stream.synchronize()

    assert app_res.shape == phase.shape == (1, 1, 1, 2)
    assert app_res.device == phase.device == device
    assert torch.isfinite(app_res).all()
    assert torch.isfinite(phase).all()

    torch.testing.assert_close(
        app_res,
        torch.full_like(app_res, rho0),
        rtol=0.08,
        atol=0.5,
    )
    expected_phase = torch.tensor([45.0, -135.0], device=device)
    assert _wrapped_phase_error(phase, expected_phase).abs().max() < 2.5
    torch.testing.assert_close(
        app_res[..., 0], app_res[..., 1], rtol=0.01, atol=0.1
    )


@pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE,
    reason="CUDA and the compiled mt_forward_cuda extension are required",
)
def test_explicit_air_halfspace_is_finite_symmetric_and_near_analytic():
    rho0 = 100.0
    device = torch.device("cuda", torch.cuda.current_device())
    rho = torch.full((1, 1, 4), rho0, dtype=torch.float64, device=device)
    solver = MTForward3D(
        freqs=[4.0],
        dx=500.0,
        dy=500.0,
        dz=125.0,
        boundary_mode="explicit_air",
        air_layers=4,
    )

    app_res, phase = solver(rho)

    assert torch.isfinite(app_res).all() and torch.isfinite(phase).all()
    torch.testing.assert_close(
        app_res, torch.full_like(app_res, rho0), rtol=0.08, atol=0.5
    )
    expected_phase = torch.tensor([45.0, -135.0], device=device)
    assert _wrapped_phase_error(phase, expected_phase).abs().max() < 2.5
    torch.testing.assert_close(
        app_res[..., 0], app_res[..., 1], rtol=0.01, atol=0.1
    )


@pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE,
    reason="CUDA and the compiled mt_forward_cuda extension are required",
)
def test_host_geometry_cache_reuses_and_rebuilds_on_key_change():
    extension = mt_module.mt_forward_cuda
    extension._clear_host_geometry_cache()
    rho_a = torch.ones((1, 1, 1), dtype=torch.float64)
    rho_b = torch.ones((2, 1, 1), dtype=torch.float64)

    first = extension.compute_mt_3d(rho_a, 100.0, 100.0, 100.0, [4.0], 100.0)
    assert all(torch.isfinite(value).all() for value in first)
    assert extension._host_geometry_cache_stats() == (0, 1)

    second = extension.compute_mt_3d(rho_a, 100.0, 100.0, 100.0, [4.0], 100.0)
    assert all(torch.isfinite(value).all() for value in second)
    hits, builds = extension._host_geometry_cache_stats()
    assert hits >= 1 and builds == 1

    third = extension.compute_mt_3d(rho_a, 120.0, 100.0, 100.0, [4.0], 100.0)
    assert all(torch.isfinite(value).all() for value in third)
    assert extension._host_geometry_cache_stats()[1] == 2

    fourth = extension.compute_mt_3d(rho_b, 100.0, 100.0, 100.0, [4.0], 100.0)
    assert all(torch.isfinite(value).all() for value in fourth)
    assert extension._host_geometry_cache_stats()[1] == 3

    fifth = extension.compute_mt_3d(rho_a, 100.0, 100.0, 100.0, [4.0], 100.0)
    assert all(torch.isfinite(value).all() for value in fifth)
    assert extension._host_geometry_cache_stats()[1] == 4

    sixth = extension.compute_mt_3d(
        rho_a,
        100.0,
        100.0,
        100.0,
        [4.0],
        100.0,
        padding_layers=12,
    )
    assert all(torch.isfinite(value).all() for value in sixth)
    assert extension._host_geometry_cache_stats()[1] == 5


@pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE,
    reason="CUDA and the compiled mt_forward_cuda extension are required",
)
def test_cpu_and_cuda_inputs_produce_consistent_outputs():
    solver = MTForward3D(freqs=[4.0, 1.0], dx=100.0, dy=80.0, dz=120.0)
    rho_cpu = torch.tensor(
        [80.0, 120.0, 200.0, 60.0], dtype=torch.float64
    ).reshape(2, 1, 2)
    rho_cuda = rho_cpu.cuda()

    app_cpu, phase_cpu = solver(rho_cpu)
    app_cuda, phase_cuda = solver(rho_cuda)

    assert app_cpu.device.type == phase_cpu.device.type == "cpu"
    assert app_cuda.device.type == phase_cuda.device.type == "cuda"
    assert app_cpu.shape == phase_cpu.shape == (2, 2, 1, 2)
    assert torch.isfinite(app_cpu).all() and torch.isfinite(phase_cpu).all()
    # Each call reassembles with floating-point atomics, so roundoff can steer
    # BiCGStab to a slightly different point within its 1e-7 residual tolerance.
    torch.testing.assert_close(app_cpu, app_cuda.cpu(), rtol=2e-4, atol=2e-4)
    phase_error = _wrapped_phase_error(phase_cpu, phase_cuda.cpu())
    assert phase_error.abs().max() < 0.02


@pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE,
    reason="CUDA and the compiled mt_forward_cuda extension are required",
)
def test_frequency_continuation_preserves_requested_output_order():
    device = torch.device("cuda", torch.cuda.current_device())
    rho = torch.tensor(
        [80.0, 120.0, 200.0, 60.0], dtype=torch.float64, device=device
    ).reshape(2, 1, 2)
    ascending = MTForward3D(freqs=[1.0, 4.0], dx=100.0, dy=80.0, dz=120.0)
    descending = MTForward3D(freqs=[4.0, 1.0], dx=100.0, dy=80.0, dz=120.0)

    app_ascending, phase_ascending = ascending(rho)
    app_descending, phase_descending = descending(rho)

    torch.testing.assert_close(
        app_ascending,
        app_descending.flip(0),
        rtol=2e-4,
        atol=2e-4,
    )
    phase_error = _wrapped_phase_error(
        phase_ascending,
        phase_descending.flip(0),
    )
    assert phase_error.abs().max() < 0.02


@pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE,
    reason="CUDA and the compiled mt_forward_cuda extension are required",
)
def test_frequency_continuation_can_be_disabled_without_changing_response():
    device = torch.device("cuda", torch.cuda.current_device())
    rho = torch.tensor(
        [80.0, 120.0, 200.0, 60.0], dtype=torch.float64, device=device
    ).reshape(2, 1, 2)
    common = dict(freqs=[4.0, 1.0], dx=100.0, dy=80.0, dz=120.0)
    warm = MTForward3D(**common, frequency_continuation=True)
    cold = MTForward3D(**common, frequency_continuation=False)

    warm_app, warm_phase = warm(rho)
    cold_app, cold_phase = cold(rho)

    torch.testing.assert_close(warm_app, cold_app, rtol=3e-4, atol=3e-4)
    phase_error = _wrapped_phase_error(warm_phase, cold_phase)
    assert phase_error.abs().max() < 0.03
