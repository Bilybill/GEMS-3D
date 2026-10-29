"""Small, repeatable wall-clock benchmark for the compiled MT extension."""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


def _parse_csv(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("mt_forward_cuda", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CUDA extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--shape", default="2,3,8")
    parser.add_argument("--freqs", default="4,1")
    parser.add_argument("--spacing", default="100,100,50")
    parser.add_argument("--rho", type=float, default=100.0)
    parser.add_argument("--air-layers", type=int, default=0)
    parser.add_argument("--air-resistivity", type=float, default=1.0e8)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--solver-max-iterations", type=int, default=50000)
    parser.add_argument("--padding-layers", type=int, default=10)
    parser.add_argument("--padding-growth", type=float, default=1.4)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--clear-geometry-cache-each-call",
        action="store_true",
        help="force a host mesh/CSR rebuild before every invocation",
    )
    args = parser.parse_args()

    shape = _parse_csv(args.shape, int)
    freqs = _parse_csv(args.freqs, float)
    spacing = _parse_csv(args.spacing, float)
    if len(shape) != 3 or len(spacing) != 3 or not freqs:
        raise ValueError("shape/spacing require three values and freqs cannot be empty")

    torch.cuda.set_device(args.device)
    extension = _load_extension(args.extension.resolve())
    rho = torch.full(shape, args.rho, dtype=torch.float64, device=args.input_device)

    timings = []
    outputs = None
    forced_cold_cache_hits = 0
    forced_cold_cache_builds = 0
    for iteration in range(args.warmup + args.repeats):
        if args.clear_geometry_cache_each_call:
            if not hasattr(extension, "_clear_host_geometry_cache"):
                raise RuntimeError("extension does not expose the geometry cache controls")
            extension._clear_host_geometry_cache()
        torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        outputs = extension.compute_mt_3d(
            rho,
            *spacing,
            freqs,
            float("nan"),
            args.air_layers,
            args.air_resistivity,
            args.solver_tolerance,
            args.solver_max_iterations,
            not args.cold_start,
            args.padding_layers,
            args.padding_growth,
        )
        torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - start
        if args.clear_geometry_cache_each_call:
            hits, builds = extension._host_geometry_cache_stats()
            forced_cold_cache_hits += hits
            forced_cold_cache_builds += builds
        if iteration >= args.warmup:
            timings.append(elapsed)

    assert outputs is not None
    app_res, phase = outputs
    cache_stats = None
    if hasattr(extension, "_host_geometry_cache_stats"):
        cache_stats = (
            (forced_cold_cache_hits, forced_cold_cache_builds)
            if args.clear_geometry_cache_each_call
            else extension._host_geometry_cache_stats()
        )
    result = {
        "shape": shape,
        "frequencies_hz": freqs,
        "input_device": args.input_device,
        "air_layers": args.air_layers,
        "solver_tolerance": args.solver_tolerance,
        "padding_layers": args.padding_layers,
        "padding_growth": args.padding_growth,
        "frequency_continuation": not args.cold_start,
        "geometry_cache_forced_cold": args.clear_geometry_cache_each_call,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_seconds": statistics.median(timings),
        "minimum_seconds": min(timings),
        "all_seconds": timings,
        "finite": bool(torch.isfinite(app_res).all() and torch.isfinite(phase).all()),
        "app_res_mean": float(app_res.mean()),
    }
    if cache_stats is not None:
        result["geometry_cache_hits"] = int(cache_stats[0])
        result["geometry_cache_builds"] = int(cache_stats[1])
    print("BENCHMARK_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
