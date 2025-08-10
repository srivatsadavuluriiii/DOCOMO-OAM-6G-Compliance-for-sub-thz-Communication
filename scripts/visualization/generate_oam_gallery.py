#!/usr/bin/env python3
"""
Generate an OAM visualization gallery for the project.

Figures produced (saved under an output directory):
- Mode comparison: phase/intensity for a list of modes
- Superpositions: combined phase/intensity for weighted mode sums
- Impairments: turbulence and pointing error effects per mode
- Propagation series: phase/intensity as beam spreads with distance
- Crosstalk matrix: annotated heatmap of inter-mode coupling
- Pair mode coupling: A + alpha*B, phase and intensity before/after

Usage example:
  python scripts/visualization/generate_oam_gallery.py \
    --modes 1 2 3 4 --out plots/oam_gallery --size 512
"""

from __future__ import annotations

import os
import sys
import math
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Ensure project root on sys.path for standalone execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def generate_oam_mode(topological_charge: int, size: int = 512, beam_waist: float = 0.30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simplified LG-like OAM field and intensity.

    Returns complex field, phase (rad), intensity (normalized to [0,1]).
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X * X + Y * Y)
    phi = np.arctan2(Y, X)

    rho = np.sqrt(2.0) * r / beam_waist
    amplitude = (rho ** abs(topological_charge)) * np.exp(-(rho ** 2) / 2.0)
    field = amplitude * np.exp(1j * topological_charge * phi)

    # Normalize field magnitude
    max_amp = np.max(np.abs(field)) or 1.0
    field = field / max_amp
    intensity = np.abs(field) ** 2
    max_i = np.max(intensity) or 1.0
    intensity = intensity / max_i
    phase = np.angle(field)
    return field, phase, intensity


def superpose(fields: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted superposition of complex fields; returns field, phase, intensity."""
    complex_fields = [f[0] for f in fields]
    w = np.asarray(weights, dtype=np.float64)
    if np.allclose(w, 0):
        w = np.ones_like(w)
    w = w / np.sqrt(np.sum(np.abs(w) ** 2))
    field = np.zeros_like(complex_fields[0], dtype=np.complex128)
    for coeff, f in zip(w, complex_fields):
        field += coeff * f
    # Normalize
    max_amp = np.max(np.abs(field)) or 1.0
    field = field / max_amp
    intensity = np.abs(field) ** 2
    intensity = intensity / (np.max(intensity) or 1.0)
    phase = np.angle(field)
    return field, phase, intensity


def apply_turbulence(field: np.ndarray, strength: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    phase_noise = strength * rng.standard_normal(field.shape)
    return field * np.exp(1j * phase_noise)


def apply_pointing_error(field: np.ndarray, shift_fraction_x: float, shift_fraction_y: float) -> np.ndarray:
    size = field.shape[0]
    dx = int(shift_fraction_x * size)
    dy = int(shift_fraction_y * size)
    return np.roll(np.roll(field, dx, axis=1), dy, axis=0)


def save_mode_comparison(modes: Sequence[int], size: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cols = len(modes)
    fig, axes = plt.subplots(2, cols, figsize=(4.0 * cols, 8.0))
    for idx, l in enumerate(modes):
        _, phase, intensity = generate_oam_mode(l, size=size)
        im0 = axes[0, idx].imshow(phase, cmap="hsv", origin="lower")
        axes[0, idx].set_title(f"OAM Mode {l} - Phase")
        if idx == cols - 1:
            plt.colorbar(im0, ax=axes[0, idx], label="Phase (rad)")
        im1 = axes[1, idx].imshow(intensity, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[1, idx].set_title(f"OAM Mode {l} - Intensity")
        if idx == cols - 1:
            plt.colorbar(im1, ax=axes[1, idx], label="Normalized Intensity")
    plt.suptitle("OAM Mode Comparison", y=0.98)
    for ax in axes.ravel():
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mode_comparison.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_superpositions(combos: Sequence[Tuple[Sequence[int], Sequence[float]]], size: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for idx, (l_vals, weights) in enumerate(combos):
        fields = [generate_oam_mode(l, size=size) for l in l_vals]
        _, phase, intensity = superpose(fields, weights)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        im0 = axs[0].imshow(phase, cmap="hsv", origin="lower")
        axs[0].set_title("OAM Superposition - Phase Pattern")
        plt.colorbar(im0, ax=axs[0], label="Phase (rad)")
        im1 = axs[1].imshow(intensity, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axs[1].set_title("OAM Superposition - Intensity Pattern")
        plt.colorbar(im1, ax=axs[1], label="Normalized Intensity")
        label = " + ".join([f"{w:.2f}×OAM{l}" for l, w in zip(l_vals, weights)])
        fig.text(0.5, 0.02, f"Superposition: {label}", ha="center", fontsize=12)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fname = f"superposition_{idx+1}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_impairments(l: int, size: int, out_dir: str, turbulence_weak: float = 0.05, turbulence_strong: float = 0.15,
                     pointing_shift: Tuple[float, float] = (0.02, 0.02)) -> None:
    os.makedirs(out_dir, exist_ok=True)
    field, phase, intensity = generate_oam_mode(l, size=size)
    field_w = apply_turbulence(field, turbulence_weak)
    field_s = apply_turbulence(field, turbulence_strong)
    field_p = apply_pointing_error(field, pointing_shift[0], pointing_shift[1])
    cases = [(phase, intensity), (np.angle(field_w), np.abs(field_w) ** 2), (np.angle(field_s), np.abs(field_s) ** 2), (np.angle(field_p), np.abs(field_p) ** 2)]
    for i in range(1, 4):
        cases[i] = (cases[i][0], cases[i][1] / (np.max(cases[i][1]) or 1.0))
    titles = ["Original", "Weak Turbulence", "Strong Turbulence", "Pointing Error"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, (ph, it) in enumerate(cases):
        im0 = axes[0, i].imshow(ph, cmap="hsv", origin="lower")
        axes[0, i].set_title(f"{titles[i]} - Phase")
        if i == 3:
            plt.colorbar(im0, ax=axes[0, i], label="Phase")
        im1 = axes[1, i].imshow(it, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[1, i].set_title(f"{titles[i]} - Intensity")
        if i == 3:
            plt.colorbar(im1, ax=axes[1, i], label="Intensity")
    for ax in axes.ravel():
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    plt.suptitle(f"Effects of Impairments on OAM Mode {l}", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"impairments_mode_{l}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_propagation_series(l: int, size: int, out_dir: str, distances: Sequence[float] = (0, 1, 2, 3, 4)) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cols = len(distances)
    fig, axes = plt.subplots(2, cols, figsize=(4.8 * cols, 8.0))
    for idx, d in enumerate(distances):
        beam_waist = 0.30 * (1.0 + 0.45 * d)  # simple spreading model
        _, phase, intensity = generate_oam_mode(l, size=size, beam_waist=beam_waist)
        im0 = axes[0, idx].imshow(phase, cmap="hsv", origin="lower")
        axes[0, idx].set_title(f"Distance = {d:.1f}")
        if idx == cols - 1:
            plt.colorbar(im0, ax=axes[0, idx], label="Phase (rad)")
        im1 = axes[1, idx].imshow(intensity, cmap="viridis", origin="lower", vmin=0, vmax=1)
        if idx == cols - 1:
            plt.colorbar(im1, ax=axes[1, idx], label="Normalized Intensity")
    axes[0, 0].set_ylabel("Phase")
    axes[1, 0].set_ylabel("Intensity")
    for ax in axes.ravel():
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    plt.suptitle(f"OAM Mode {l} Propagation", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"propagation_mode_{l}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_crosstalk_matrix(modes: Sequence[int], distance_factor: float = 0.5) -> np.ndarray:
    """Synthetic crosstalk: diagonal 1, off-diagonals decay with |Δl| and scale with distance."""
    n = len(modes)
    M = np.zeros((n, n), dtype=np.float64)
    for i, li in enumerate(modes):
        for j, lj in enumerate(modes):
            if i == j:
                M[i, j] = 1.0
            else:
                delta = abs(li - lj)
                M[i, j] = 0.12 * distance_factor / max(delta, 1)
    return M


def save_crosstalk_matrix(modes: Sequence[int], out_dir: str, distance_factor: float = 0.5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    M = compute_crosstalk_matrix(modes, distance_factor=distance_factor)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Coupling Strength")
    ax.set_title(f"OAM Mode Crosstalk Matrix (Distance Factor: {distance_factor})")
    ax.set_xlabel("Receiving OAM Mode")
    ax.set_ylabel("Transmitting OAM Mode")
    ax.set_xticks(range(len(modes)))
    ax.set_yticks(range(len(modes)))
    ax.set_xticklabels([str(m) for m in modes])
    ax.set_yticklabels([str(m) for m in modes])
    for i in range(len(modes)):
        for j in range(len(modes)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] > 0.5 else "w")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "crosstalk_matrix.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pair_coupling(pairs: Sequence[Tuple[int, int]], alpha: float, size: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for a, b in pairs:
        fa, pha, inta = generate_oam_mode(a, size=size)
        fb, phb, intb = generate_oam_mode(b, size=size)
        f_c = fa + alpha * fb
        max_amp = np.max(np.abs(f_c)) or 1.0
        f_c = f_c / max_amp
        phc = np.angle(f_c)
        inc = np.abs(f_c) ** 2
        inc = inc / (np.max(inc) or 1.0)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        im00 = axes[0, 0].imshow(pha, cmap="hsv", origin="lower")
        axes[0, 0].set_title(f"OAM Mode {a} - Phase")
        im10 = axes[1, 0].imshow(inta, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[1, 0].set_title(f"OAM Mode {a} - Intensity")
        im01 = axes[0, 1].imshow(phb, cmap="hsv", origin="lower")
        axes[0, 1].set_title(f"OAM Mode {b} - Phase")
        im11 = axes[1, 1].imshow(intb, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[1, 1].set_title(f"OAM Mode {b} - Intensity")
        im02 = axes[0, 2].imshow(phc, cmap="hsv", origin="lower")
        axes[0, 2].set_title("Coupled Mode - Phase")
        im12 = axes[1, 2].imshow(inc, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[1, 2].set_title("Coupled Mode - Intensity")
        plt.colorbar(im00, ax=axes[0, 0], label="Phase")
        plt.colorbar(im12, ax=axes[1, 2], label="Normalized Intensity")
        plt.suptitle(f"Mode Coupling: OAM{a} + {alpha:.2f}×OAM{b}", y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"coupling_{a}_{b}.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def main(argv: List[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate OAM visualization gallery")
    parser.add_argument("--modes", nargs="*", type=int, default=[1, 2, 3, 4], help="OAM modes to include")
    parser.add_argument("--size", type=int, default=512, help="Grid size in pixels")
    parser.add_argument("--out", type=str, default="plots/oam_gallery", help="Output directory")
    parser.add_argument("--no-all", action="store_true", help="Do not generate all figures, only selected ones")
    parser.add_argument("--only", choices=[
        "comparison", "superpositions", "impairments", "propagation", "crosstalk", "coupling"
    ], nargs="*", help="Subset of figures to generate")

    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    targets = set(args.only or ([] if args.no_all else [
        "comparison", "superpositions", "impairments", "propagation", "crosstalk", "coupling"
    ]))

    if "comparison" in targets:
        save_mode_comparison(args.modes, size=args.size, out_dir=args.out)

    if "superpositions" in targets:
        combos = [
            (([1, 2], [1.0, 0.5])),
            (([2, 3], [1.0, 0.5])),
            (([3, 4], [1.0, 0.5])),
            (([4, 5], [1.0, 0.5])),
            (([1, 3, 4], [0.7, 0.5, 0.3])),
            (([1, 4, 6], [0.7, 0.5, 0.3])),
        ]
        save_superpositions([c for c in combos if set(c[0]).issubset(set(range(1, 10)))], size=args.size, out_dir=args.out)

    if "impairments" in targets:
        for l in [m for m in args.modes if 1 <= m <= 8][:3]:  # limit to three modes for speed
            save_impairments(l, size=args.size, out_dir=args.out)

    if "propagation" in targets:
        for l in args.modes[:3]:
            save_propagation_series(l, size=args.size, out_dir=args.out)

    if "crosstalk" in targets:
        save_crosstalk_matrix(args.modes, out_dir=args.out, distance_factor=0.5)

    if "coupling" in targets:
        pairs = []
        for i in range(len(args.modes) - 1):
            pairs.append((args.modes[i], args.modes[i + 1]))
        if pairs:
            save_pair_coupling(pairs, alpha=0.30, size=args.size, out_dir=args.out)

    print(f"Gallery written to: {args.out}")


if __name__ == "__main__":
    main()


