#!/usr/bin/env python
"""Check C-space key configuration quality against 4 criteria.

Requirements:
1. mesh_collision_ratio should NOT be close to 0 or 1
2. unsafe_ratio should be greater than mesh_collision_ratio
3. signed_clearance_norm should have safe, near-boundary, and unsafe points
4. near_boundary_ratio should not be too low

Reads artifacts from the stage-3 pipeline output:
  analysis_outputs/workpiece_key_config_collision_features/
"""

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


DEFAULT_INPUT_DIR = "analysis_outputs/workpiece_key_config_collision_features"
DEFAULT_OUTPUT_DIR = "analysis_outputs/cspace_key_config_check"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=str, default=DEFAULT_INPUT_DIR,
        help="Directory containing stage-3 collision feature artifacts.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save check report and plots.",
    )
    parser.add_argument(
        "--mesh-collision-low", type=float, default=0.05,
        help="Lower bound for mesh_collision_ratio (default: 0.05).",
    )
    parser.add_argument(
        "--mesh-collision-high", type=float, default=0.95,
        help="Upper bound for mesh_collision_ratio (default: 0.95).",
    )
    parser.add_argument(
        "--unsafe-margin", type=float, default=0.01,
        help="Minimum gap: unsafe_ratio - mesh_collision_ratio (default: 0.01).",
    )
    parser.add_argument(
        "--clearance-near-m", type=float, default=0.001,
        help="|d_min - d_safe| threshold (m) for near-boundary (default: 0.001).",
    )
    parser.add_argument(
        "--clearance-min-type-ratio", type=float, default=0.05,
        help="Min ratio per clearance type (safe/near/unsafe) for criterion 3 (default: 0.05).",
    )
    parser.add_argument(
        "--near-boundary-min", type=float, default=0.05,
        help="Minimum near_boundary_ratio for criterion 4 (default: 0.05).",
    )
    return parser


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(input_dir: pathlib.Path) -> dict:
    features_path = input_dir / "workpiece_key_config_features.npy"
    manifest_path = input_dir / "manifest.json"
    missing = [str(p) for p in (features_path, manifest_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {input_dir}: {missing}")

    features = np.asarray(np.load(features_path), dtype=np.float32)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if features.ndim != 3 or features.shape[2] != 2:
        raise ValueError(f"Expected features shape (W, K, 2), got {features.shape}")

    ids_path = input_dir / "workpiece_ids.npy"
    types_path = input_dir / "workpiece_types.npy"
    names_path = input_dir / "workpiece_names.npy"

    return {
        "features": features,
        "manifest": manifest,
        "workpiece_ids": np.load(ids_path) if ids_path.exists() else None,
        "workpiece_types": np.load(types_path) if types_path.exists() else None,
        "workpiece_names": np.load(names_path) if names_path.exists() else None,
    }


def compute_metrics(
    features: np.ndarray,
    d_safe: float,
    mesh_collision_low: float,
    mesh_collision_high: float,
    unsafe_margin: float,
    clearance_near_m: float,
    clearance_min_type_ratio: float,
    near_boundary_min: float,
) -> dict:
    num_wp, num_kc, _ = features.shape
    collision_flags = features[:, :, 0].astype(np.float32)
    d_min = features[:, :, 1]

    # --- 1. mesh_collision_ratio ---
    mesh_col_ratio_wp = collision_flags.mean(axis=1)
    mesh_col_ratio_global = float(collision_flags.mean())

    # --- 2. unsafe_ratio = mesh_collision OR (d_min <= d_safe) ---
    unsafe = (collision_flags > 0.5) | (d_min <= d_safe)
    unsafe_ratio_wp = unsafe.mean(axis=1).astype(np.float64)
    unsafe_ratio_global = float(unsafe.mean())

    # --- 3 & 4. signed clearance types ---
    signed_clearance = d_min - d_safe
    near = np.abs(signed_clearance) <= clearance_near_m
    safe = signed_clearance > clearance_near_m
    unsafe_sdf = signed_clearance < -clearance_near_m

    safe_ratio_wp = safe.mean(axis=1)
    near_ratio_wp = near.mean(axis=1)
    unsafe_sdf_ratio_wp = unsafe_sdf.mean(axis=1)

    safe_ratio_global = float(safe.mean())
    near_ratio_global = float(near.mean())
    unsafe_sdf_ratio_global = float(unsafe_sdf.mean())

    # --- Checks ---

    # Check 1
    mesh_ok_wp = (mesh_col_ratio_wp > mesh_collision_low) & (mesh_col_ratio_wp < mesh_collision_high)
    mesh_ok_global = bool(mesh_collision_low < mesh_col_ratio_global < mesh_collision_high)

    # Check 2
    gap_wp = unsafe_ratio_wp - mesh_col_ratio_wp
    unsafe_gt_mesh_wp = gap_wp >= unsafe_margin
    unsafe_gt_mesh_global = bool(unsafe_ratio_global - mesh_col_ratio_global >= unsafe_margin)

    # Check 3
    has_safe_wp = safe_ratio_wp >= clearance_min_type_ratio
    has_near_wp = near_ratio_wp >= clearance_min_type_ratio
    has_unsafe_wp = unsafe_sdf_ratio_wp >= clearance_min_type_ratio
    three_types_wp = has_safe_wp & has_near_wp & has_unsafe_wp

    has_safe_global = safe_ratio_global >= clearance_min_type_ratio
    has_near_global = near_ratio_global >= clearance_min_type_ratio
    has_unsafe_global = unsafe_sdf_ratio_global >= clearance_min_type_ratio
    three_types_global = has_safe_global and has_near_global and has_unsafe_global

    # Check 4
    near_ok_wp = near_ratio_wp >= near_boundary_min
    near_ok_global = bool(near_ratio_global >= near_boundary_min)

    # --- Build result ---
    def _l(arr):
        return arr.astype(float).tolist()

    return {
        "num_workpieces": num_wp,
        "num_key_configs": num_kc,
        "d_safe_m": d_safe,
        "thresholds": {
            "mesh_collision_low": mesh_collision_low,
            "mesh_collision_high": mesh_collision_high,
            "unsafe_margin": unsafe_margin,
            "clearance_near_m": clearance_near_m,
            "clearance_min_type_ratio": clearance_min_type_ratio,
            "near_boundary_min": near_boundary_min,
        },
        "global": {
            "mesh_collision_ratio": float(mesh_col_ratio_global),
            "unsafe_ratio": float(unsafe_ratio_global),
            "unsafe_ratio_gap": float(unsafe_ratio_global - mesh_col_ratio_global),
            "safe_ratio": float(safe_ratio_global),
            "near_boundary_ratio": float(near_ratio_global),
            "unsafe_sdf_ratio": float(unsafe_sdf_ratio_global),
            "signed_clearance_stats_m": {
                "min": float(signed_clearance.min()),
                "max": float(signed_clearance.max()),
                "mean": float(signed_clearance.mean()),
                "std": float(signed_clearance.std()),
                "p5": float(np.percentile(signed_clearance, 5)),
                "p25": float(np.percentile(signed_clearance, 25)),
                "p50": float(np.percentile(signed_clearance, 50)),
                "p75": float(np.percentile(signed_clearance, 75)),
                "p95": float(np.percentile(signed_clearance, 95)),
            },
        },
        "checks": {
            "1_mesh_collision_ratio_not_extreme": {
                "pass": mesh_ok_global,
                "global_value": float(mesh_col_ratio_global),
                "threshold": f"({mesh_collision_low}, {mesh_collision_high})",
                "pass_rate_per_workpiece": float(mesh_ok_wp.mean()),
                "failed_workpiece_indices": _l(np.flatnonzero(~mesh_ok_wp)),
            },
            "2_unsafe_ratio_gt_mesh_collision_ratio（基于 Dim 2 safety_flag）": {
                "pass": unsafe_gt_mesh_global,
                "global_gap": float(unsafe_ratio_global - mesh_col_ratio_global),
                "threshold": f"gap >= {unsafe_margin}",
                "pass_rate_per_workpiece": float(unsafe_gt_mesh_wp.mean()),
                "failed_workpiece_indices": _l(np.flatnonzero(~unsafe_gt_mesh_wp)),
            },
            "3_signed_clearance_three_types": {
                "pass": three_types_global,
                "has_safe": has_safe_global,
                "has_near_boundary": has_near_global,
                "has_unsafe": has_unsafe_global,
                "safe_ratio": float(safe_ratio_global),
                "near_boundary_ratio": float(near_ratio_global),
                "unsafe_sdf_ratio": float(unsafe_sdf_ratio_global),
                "threshold_per_type": f">= {clearance_min_type_ratio}",
                "pass_rate_per_workpiece": float(three_types_wp.mean()),
                "failed_workpiece_indices": _l(np.flatnonzero(~three_types_wp)),
            },
            "4_near_boundary_ratio_not_too_low": {
                "pass": near_ok_global,
                "global_value": float(near_ratio_global),
                "threshold": f">= {near_boundary_min}",
                "pass_rate_per_workpiece": float(near_ok_wp.mean()),
                "failed_workpiece_indices": _l(np.flatnonzero(~near_ok_wp)),
            },
        },
        "per_workpiece": {
            "mesh_collision_ratio": _l(mesh_col_ratio_wp),
            "unsafe_ratio": _l(unsafe_ratio_wp),
            "safe_ratio": _l(safe_ratio_wp),
            "near_boundary_ratio": _l(near_ratio_wp),
            "unsafe_sdf_ratio": _l(unsafe_sdf_ratio_wp),
            "all_checks_pass": _l(mesh_ok_wp & unsafe_gt_mesh_wp & three_types_wp & near_ok_wp),
        },
    }


def plot_report(results: dict, workpiece_types: Optional[np.ndarray], output_dir: pathlib.Path) -> None:
    if not HAS_MPL:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    global_ = results["global"]
    per_wp = results["per_workpiece"]
    checks = results["checks"]
    thresholds = results["thresholds"]
    num_wp = results["num_workpieces"]

    # Colors per workpiece type
    if workpiece_types is not None:
        types = np.asarray(workpiece_types).astype(str)
        type_color_map = {"regular": "tab:blue", "simple": "tab:orange"}
        wp_colors = [type_color_map.get(t, "tab:gray") for t in types]
    else:
        wp_colors = ["tab:blue"] * num_wp

    x = np.arange(num_wp)

    # ---- Per-workpiece 4-panel ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Panel 1: mesh vs unsafe ratio
    ax = axes[0, 0]
    ax.bar(x - 0.15, per_wp["mesh_collision_ratio"], width=0.3, label="mesh_collision",
           color=wp_colors, alpha=0.7)
    ax.bar(x + 0.15, per_wp["unsafe_ratio"], width=0.3, label="unsafe",
           color=wp_colors, alpha=0.4, edgecolor="black")
    ax.axhline(global_["mesh_collision_ratio"], color="blue", ls="--", lw=0.8, label="global mesh")
    ax.axhline(global_["unsafe_ratio"], color="red", ls=":", lw=0.8, label="global unsafe")
    ax.set_title("1. mesh_collision_ratio vs unsafe_ratio")
    ax.set_xlabel("workpiece index")
    ax.set_ylabel("ratio")
    ax.legend(fontsize=7)

    # Panel 2: unsafe gap
    ax = axes[0, 1]
    gaps = np.array(per_wp["unsafe_ratio"]) - np.array(per_wp["mesh_collision_ratio"])
    ax.bar(x, gaps, color=wp_colors, alpha=0.7)
    ax.axhline(thresholds["unsafe_margin"], color="red", ls="--", lw=0.8, label="min margin")
    ax.set_title("2. unsafe_ratio - mesh_collision_ratio gap")
    ax.set_xlabel("workpiece index")
    ax.set_ylabel("gap")
    ax.legend(fontsize=7)

    # Panel 3: clearance type stacked bars
    ax = axes[1, 0]
    safe_arr = np.array(per_wp["safe_ratio"])
    near_arr = np.array(per_wp["near_boundary_ratio"])
    unsafe_arr = np.array(per_wp["unsafe_sdf_ratio"])
    ax.bar(x, safe_arr, label="safe", color="tab:green", alpha=0.7)
    ax.bar(x, near_arr, bottom=safe_arr, label="near boundary", color="tab:orange", alpha=0.7)
    ax.bar(x, unsafe_arr, bottom=safe_arr + near_arr, label="unsafe (SDF)", color="tab:red", alpha=0.7)
    ax.set_title("3 & 4. Clearance type distribution")
    ax.set_xlabel("workpiece index")
    ax.set_ylabel("ratio")
    ax.legend(fontsize=7)

    # Panel 4: per-check pass/fail heatmap
    ax = axes[1, 1]
    mesh_wp = np.array(per_wp["mesh_collision_ratio"])
    unsafe_wp = np.array(per_wp["unsafe_ratio"])
    near_wp = np.array(per_wp["near_boundary_ratio"])
    safe_wp_arr = np.array(per_wp["safe_ratio"])
    unsafe_sdf_wp = np.array(per_wp["unsafe_sdf_ratio"])

    t = thresholds
    check_mat = np.zeros((num_wp, 4), dtype=float)
    check_mat[:, 0] = (mesh_wp > t["mesh_collision_low"]) & (mesh_wp < t["mesh_collision_high"])
    check_mat[:, 1] = (unsafe_wp - mesh_wp) >= t["unsafe_margin"]
    check_mat[:, 2] = (
        (safe_wp_arr >= t["clearance_min_type_ratio"])
        & (near_wp >= t["clearance_min_type_ratio"])
        & (unsafe_sdf_wp >= t["clearance_min_type_ratio"])
    )
    check_mat[:, 3] = near_wp >= t["near_boundary_min"]

    check_labels = ["1.mesh_not_extreme", "2.unsafe_gt_mesh", "3.three_types", "4.near_boundary"]
    im = ax.imshow(check_mat.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels(check_labels, fontsize=7)
    ax.set_xlabel("workpiece index")
    ax.set_title("Per-check pass/fail (green=pass, red=fail)")
    plt.colorbar(im, ax=ax, ticks=[0, 1])

    fig.tight_layout()
    fig.savefig(output_dir / "per_workpiece_report.pdf", dpi=150)
    fig.savefig(output_dir / "per_workpiece_report.png", dpi=150)
    plt.close(fig)

    # ---- Summary plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Category bar
    ax1.bar(["safe", "near boundary", "unsafe (SDF)"],
            [global_["safe_ratio"], global_["near_boundary_ratio"], global_["unsafe_sdf_ratio"]],
            color=["tab:green", "tab:orange", "tab:red"], alpha=0.7)
    ax1.set_title("Global clearance type distribution")
    ax1.set_ylabel("ratio")

    # Checklist text
    ax2.axis("off")
    lines = ["=== C-space Key Configuration Check ===", ""]
    for name, c in checks.items():
        status = "PASS" if c["pass"] else "FAIL"
        lines.append(f"  [{status}] {name}")
    lines.append("")
    sc = global_["signed_clearance_stats_m"]
    lines.append(f"  mesh_collision_ratio  = {global_['mesh_collision_ratio']:.4f}")
    lines.append(f"  unsafe_ratio          = {global_['unsafe_ratio']:.4f}")
    lines.append(f"  near_boundary_ratio   = {global_['near_boundary_ratio']:.4f}")
    lines.append(f"  signed_clearance      = {sc['mean']:.6f} +/- {sc['std']:.6f}")
    lines.append(f"  d_safe                = {results['d_safe_m']} m")
    lines.append(f"  workpieces × configs  = {results['num_workpieces']} × {results['num_key_configs']}")
    ax2.text(0, 0.5, "\n".join(lines), fontfamily="monospace", fontsize=9, va="center")

    fig.tight_layout()
    fig.savefig(output_dir / "summary_report.pdf", dpi=150)
    fig.savefig(output_dir / "summary_report.png", dpi=150)
    plt.close(fig)


def print_simple_report(results: dict) -> None:
    checks = results["checks"]
    global_ = results["global"]

    all_pass = all(c["pass"] for c in checks.values())

    print()
    print("=" * 62)
    print("  C-space Key Configuration Quality Check")
    print("=" * 62)
    print(f"  Data: {results['num_workpieces']} workpieces × {results['num_key_configs']} key configs")
    print(f"  d_safe = {results['d_safe_m']} m")
    print()

    for check_name, c in checks.items():
        status = "✓ PASS" if c["pass"] else "✗ FAIL"
        print(f"  {status}  {check_name}")
        if "global_value" in c:
            print(f"           value={c['global_value']:.4f}  threshold={c['threshold']}")
        elif "global_gap" in c:
            print(f"           gap={c['global_gap']:.4f}  threshold={c['threshold']}")
        elif "has_safe" in c:
            print(f"           safe={c['has_safe']}  near={c['has_near_boundary']}  unsafe={c['has_unsafe']}")
            print(f"           ratios: safe={c['safe_ratio']:.4f} near={c['near_boundary_ratio']:.4f} unsafe={c['unsafe_sdf_ratio']:.4f}")
        if c["pass_rate_per_workpiece"] < 1.0:
            n_fail = len(c.get("failed_workpiece_indices", []))
            print(f"           {n_fail}/{results['num_workpieces']} workpieces fail")
            if n_fail <= 10 and "failed_workpiece_indices" in c:
                print(f"           failed indices: {c['failed_workpiece_indices']}")
        print()

    print("  Global metrics:")
    print(f"    mesh_collision_ratio      = {global_['mesh_collision_ratio']:.4f}")
    print(f"    unsafe_ratio              = {global_['unsafe_ratio']:.4f}")
    print(f"    unsafe gap                = {global_['unsafe_ratio_gap']:.4f}")
    print(f"    near_boundary_ratio       = {global_['near_boundary_ratio']:.4f}")
    print()
    sc = global_["signed_clearance_stats_m"]
    print("    signed_clearance (d_min - d_safe) [m]:")
    print(f"      min={sc['min']:.6f}  p5={sc['p5']:.6f}  p25={sc['p25']:.6f}  p50={sc['p50']:.6f}")
    print(f"      p75={sc['p75']:.6f}  p95={sc['p95']:.6f}  max={sc['max']:.6f}")
    print(f"      mean={sc['mean']:.6f}  std={sc['std']:.6f}")
    print()

    print("  " + "-" * 56)
    if all_pass:
        print("  ✓ ALL CHECKS PASSED")
    else:
        n_fail = sum(1 for c in checks.values() if not c["pass"])
        print(f"  ✗ {n_fail}/4 CHECKS FAILED")
    print("=" * 62)


def main() -> None:
    args = build_parser().parse_args()
    input_dir = pathlib.Path(args.input_dir).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        print(
            "Run stage 1-3 pipeline first:\n"
            "  python scripts/build_joint_configuration_pool.py\n"
            "  python scripts/select_key_joint_configurations_fps.py\n"
            "  python scripts/build_workpiece_key_config_collision_features.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading data from: {input_dir}")
    data = load_data(input_dir)

    manifest = data["manifest"]
    if "d_safe_m" not in manifest:
        raise KeyError("manifest.json missing 'd_safe_m' field")
    d_safe = float(manifest["d_safe_m"])

    print(f"Computing metrics (d_safe = {d_safe} m)...\n")
    results = compute_metrics(
        features=data["features"],
        d_safe=d_safe,
        mesh_collision_low=float(args.mesh_collision_low),
        mesh_collision_high=float(args.mesh_collision_high),
        unsafe_margin=float(args.unsafe_margin),
        clearance_near_m=float(args.clearance_near_m),
        clearance_min_type_ratio=float(args.clearance_min_type_ratio),
        near_boundary_min=float(args.near_boundary_min),
    )

    # Terminal report
    print_simple_report(results)

    # JSON report
    reporter_path = output_dir / "check_report.json"
    with open(reporter_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON report saved to: {reporter_path}")

    # Plots
    if HAS_MPL:
        plot_report(results, data["workpiece_types"], output_dir)
        print(f"Plots saved to: {output_dir}")
    else:
        print("[WARN] matplotlib not available, skipping plots. `pip install matplotlib` to enable.")


if __name__ == "__main__":
    main()
