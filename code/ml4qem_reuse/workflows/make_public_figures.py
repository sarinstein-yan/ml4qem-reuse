#!/usr/bin/env python3
"""Generate the public figures from frozen JSON/NPZ evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4qem_reuse.workflows._paths import output_root, workspace_root
from ml4qem_reuse.workflows.train_local_within_domain import LocalData, _aggregate_by_base


PROJECT = workspace_root()
PUBLIC_OUTPUT = output_root()
FIGURES = PUBLIC_OUTPUT / "figures"
RESULTS = PROJECT / "results"

NAVY = "#243B53"
BLUE = "#3973AC"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
GREY = "#7A8793"
LIGHT = "#E8EEF3"
FROZEN_FIGURE_TIME = datetime(2026, 8, 15, tzinfo=timezone.utc)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": True,
        }
    )


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=NAVY,
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("pdf", 300), ("png", 360)):
        metadata = (
            {
                "Creator": "ML4QEM reusability report",
                "Producer": "Matplotlib",
                "CreationDate": FROZEN_FIGURE_TIME,
                "ModDate": FROZEN_FIGURE_TIME,
            }
            if suffix == "pdf"
            else {"Software": "ML4QEM reusability report"}
        )
        fig.savefig(
            FIGURES / f"{name}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.025,
            transparent=True,
            metadata=metadata,
        )
    plt.close(fig)


def figure1(summary: dict[str, object]) -> None:
    headline = summary["headline"]
    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.05, 1), hspace=0.42)
    ax = fig.add_subplot(grid[0])
    ax.axis("off")
    panel(ax, "a")
    labels = [
        ("Public artifact", "DOI snapshot\nsource data + notebooks"),
        ("Pinned legacy", "paper-era stack\none-line code repair"),
        ("Current stack", "portable encoder\nRF equivalence"),
        ("Reuse tests", "grouped shifts\nglobal raw anchor"),
    ]
    xs = np.linspace(0.03, 0.79, len(labels))
    width = 0.18
    for index, (title, body) in enumerate(labels):
        color = (LIGHT, "#DCEDE9", "#E5EBF6", "#F9E9E3")[index]
        box = mpl.patches.FancyBboxPatch(
            (xs[index], 0.24),
            width,
            0.53,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            facecolor=color,
            edgecolor=NAVY,
            linewidth=0.9,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(xs[index] + width / 2, 0.62, title, ha="center", fontweight="bold", transform=ax.transAxes)
        ax.text(xs[index] + width / 2, 0.40, body, ha="center", va="center", transform=ax.transAxes, linespacing=1.3)
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1] - 0.012, 0.505),
                xytext=(xs[index] + width + 0.012, 0.505),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": 1.2},
            )
    ax.text(0.99, 0.04, "No new QPU jobs", ha="right", color=ORANGE, fontweight="bold", transform=ax.transAxes)

    bottom = grid[1].subgridspec(1, 4, wspace=0.25)
    cards = [
        ("Cold start", "8/9", "tests passed"),
        ("Minimal repair", "9/9", "tests passed"),
        ("Encoder", f"{headline['portability']['feature_max_abs_difference']:.2g}", "maximum difference"),
        (
            "RF portability",
            f"{max(headline['portability']['demo1_prediction_max_abs_difference'], headline['portability']['demo2_prediction_max_abs_difference']):.1e}",
            "maximum prediction\ndifference",
        ),
    ]
    for index, (title, number, caption) in enumerate(cards):
        card = fig.add_subplot(bottom[index])
        card.axis("off")
        if index == 0:
            panel(card, "b")
        patch = mpl.patches.FancyBboxPatch(
            (0.02, 0.04),
            0.96,
            0.90,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor="white",
            edgecolor=LIGHT,
            linewidth=1.1,
            transform=card.transAxes,
        )
        card.add_patch(patch)
        card.text(0.5, 0.76, title, ha="center", color=NAVY, fontweight="bold", transform=card.transAxes)
        card.text(0.5, 0.48, number, ha="center", va="center", fontsize=15, color=(ORANGE if index == 0 else TEAL), fontweight="bold", transform=card.transAxes)
        card.text(0.5, 0.18, caption, ha="center", color=GREY, transform=card.transAxes, linespacing=1.25)
    save(fig, "fig1_reuse_path")


def figure2(summary: dict[str, object]) -> None:
    audit = load_json(RESULTS / "development/published_result_audit.json")
    hardware = summary["headline"]["archived_hardware"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    ax = axes[0, 0]
    panel(ax, "a")
    order = ["Unmitigated", "ZNE", "OLS", "RF", "MLP", "GNN"]
    values = [audit["figure2"]["comparisons"][name]["source"] for name in order]
    colors = [GREY, GOLD, BLUE, TEAL, "#7768AE", ORANGE]
    ax.bar(np.arange(len(order)), values, color=colors, width=0.72)
    ax.set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
    ax.set_ylabel("Mean L₂ vector error")
    ax.set_title("Distributed source-data means recovered")
    ax.text(0.98, 0.95, "max |difference| < 10⁻¹⁵", ha="right", va="top", transform=ax.transAxes, color=TEAL)

    ax = axes[0, 1]
    panel(ax, "b")
    methods = ["noisy", "zne", "rf"]
    x = np.arange(3)
    ax.bar(x - 0.18, [hardware["metrics"][m]["mae"] for m in methods], width=0.36, label="MAE", color=BLUE)
    ax.bar(x + 0.18, [hardware["metrics"][m]["p95_absolute_error"] for m in methods], width=0.36, label="95th percentile", color=ORANGE)
    ax.set_xticks(x, ["Unmitigated", "ZNE", "RF"])
    ax.set_ylabel("Absolute error")
    ax.set_title("Archived ibm_algiers reanalysis")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    panel(ax, "c")
    with np.load(RESULTS / "confirmation/archived_hardware_reanalysis_predictions.npz", allow_pickle=False) as archive:
        depth = archive["depth"]
        target = archive["target"]
        for method, label, color in (
            ("noisy", "Unmitigated", GREY),
            ("zne", "ZNE", GOLD),
            ("rf", "RF", TEAL),
        ):
            error = np.abs(archive[f"{method}__prediction"] - target)
            y = [np.mean(error[depth == value]) for value in np.unique(depth)]
            ax.plot(np.unique(depth), y, marker="o", ms=3, lw=1.4, label=label, color=color)
    ax.set_xlabel("Depth step")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("The archived RF advantage persists across depth")
    ax.legend(frameon=False, ncol=3)

    ax = axes[1, 1]
    panel(ax, "d")
    resources = hardware["resources"]
    x = np.arange(2)
    circuits = np.asarray([resources["zne_total_circuit_executions_reported"], resources["ml_qem_total_circuit_executions_reported"]]) / 1000
    shots = np.asarray([resources["zne_total_shots_equivalent"], resources["ml_qem_total_shots_equivalent"]]) / 1e6
    ax.bar(x - 0.18, circuits, width=0.36, color=BLUE, label="Circuits (thousands)")
    twin = ax.twinx()
    twin.bar(x + 0.18, shots, width=0.36, color=ORANGE, label="Shots (millions)")
    ax.set_xticks(x, ["ZNE", "ML4QEM"])
    ax.set_ylabel("Circuit executions (×10³)", color=BLUE)
    twin.set_ylabel("Shot equivalent (×10⁶)", color=ORANGE)
    ax.set_title("Historical execution accounting")
    ax.text(0.98, 0.94, "40% fewer total circuits", ha="right", va="top", transform=ax.transAxes, color=TEAL, fontweight="bold")
    save(fig, "fig2_public_and_archived_evidence")


def figure3(summary: dict[str, object]) -> None:
    curve = load_json(RESULTS / "confirmation/v2/training_size_curve.json")
    within = load_json(RESULTS / "confirmation/v2/local_within_n512_folds0-4_seed0.json")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    ax = axes[0, 0]
    panel(ax, "a")
    ax.axis("off")
    cards = [
        ("4", "circuit families"),
        ("3 × 3", "noise families × strengths"),
        ("4 × 3", "shot budgets × replicates"),
        ("4", "Z observables"),
    ]
    for index, (number, label) in enumerate(cards):
        row, col = divmod(index, 2)
        x, y = 0.02 + col * 0.50, 0.51 - row * 0.48
        box = mpl.patches.FancyBboxPatch((x, y), 0.45, 0.39, boxstyle="round,pad=0.015", facecolor=(LIGHT if index % 2 == 0 else "#DCEDE9"), edgecolor="none", transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + 0.225, y + 0.25, number, ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY, transform=ax.transAxes)
        wrapped_label = label.replace(" × ", " ×\n")
        ax.text(
            x + 0.225,
            y + 0.10,
            wrapped_label,
            ha="center",
            va="center",
            color=GREY,
            linespacing=1.05,
            transform=ax.transAxes,
        )
    ax.set_title("960 base circuits; descendants never cross splits")

    ax = axes[0, 1]
    panel(ax, "b")
    sizes = []
    effect, low, high = [], [], []
    for cell in curve["cells"]:
        item = cell["paired_mae_effect_vs_noisy"]["safe_simplex"]
        sizes.append(cell["training_size"])
        effect.append(item["estimate"])
        low.append(item["ci_low"])
        high.append(item["ci_high"])
    effect = np.asarray(effect)
    ax.errorbar(sizes, effect, yerr=[effect - low, np.asarray(high) - effect], fmt="o-", color=TEAL, ecolor=TEAL, capsize=2.5, lw=1.4)
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.fill_between([0, 600], 0, 0.008, color="#FEF0EA", alpha=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(value) for value in sizes])
    ax.set_xlim(25, 600)
    ax.set_xlabel("Base circuits available to training partition")
    ax.set_ylabel("Paired MAE difference vs unmitigated")
    ax.set_title("Clear mean benefit only at the largest tested size")

    ax = axes[1, 0]
    panel(ax, "c")
    methods = [
        "noisy_sampled",
        "rf",
        "simplex",
        "safe_simplex",
        "raw_cell_affine",
        "safe_cell_simplex",
    ]
    labels = [
        "Unmitigated",
        "RF",
        "4-model\nsimplex",
        "Global\nanchor",
        "Matched\ncontrol",
        "Tier-cell\ndiagnostic",
    ]
    values = [within["raw_metrics"][method]["mae"] for method in methods]
    ax.bar(np.arange(len(methods)), values, color=[GREY, BLUE, GOLD, "#62B6A7", "#A7A9AC", TEAL])
    ax.set_xticks(np.arange(len(methods)), labels)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Confirmation on all five v2 folds")

    ax = axes[1, 1]
    panel(ax, "d")
    weight_arrays = []
    for record in within["fold_records"]:
        weight_arrays.append(record["safe_cell_simplex_weights"])
    shots = [128, 512, 2048, 10000]
    anchor = np.empty((3, 4))
    for strength in range(3):
        for shot_index, shot in enumerate(shots):
            key = f"S{strength}:N{shot}"
            anchor[strength, shot_index] = np.mean([weights[key][0] for weights in weight_arrays])
    image = ax.imshow(anchor, vmin=0, vmax=1, cmap=mpl.colors.LinearSegmentedColormap.from_list("anchor", ["white", TEAL]), aspect="auto")
    ax.set_xticks(range(4), ["128", "512", "2,048", "10,000"])
    ax.set_yticks(range(3), ["low", "medium", "high"])
    ax.set_xlabel("Shot budget")
    ax.set_ylabel("Noise strength")
    ax.set_title("Tier-cell diagnostic: unmitigated-expert weight")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{anchor[i, j]:.2f}", ha="center", va="center", color=("white" if anchor[i, j] > 0.55 else NAVY), fontsize=6.8)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    save(fig, "fig3_controlled_reuse_test")


def figure4(summary: dict[str, object]) -> None:
    within = load_json(RESULTS / "confirmation/v2/local_within_n512_folds0-4_seed0.json")
    with np.load(
        PROJECT / "data/derived/local_benchmark_confirmation_v2.npz", allow_pickle=False
    ) as source, np.load(
        RESULTS / "confirmation/v2/local_within_n512_folds0-4_seed0_predictions.npz",
        allow_pickle=False,
    ) as predictions:
        data = LocalData(source)
        rows = predictions["row_index"]
        base = data.row_base[rows]
        target = predictions["target"]
        noisy = predictions["raw__noisy_sampled__prediction"]
        safe = predictions["raw__safe_simplex__prediction"]
        noisy_base = _aggregate_by_base(np.mean(np.abs(noisy - target), axis=1), base)
        safe_base = _aggregate_by_base(np.mean(np.abs(safe - target), axis=1), base)
        paired = safe_base - noisy_base
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    panel(ax, "a")
    ax.hist(paired, bins=34, color=TEAL, alpha=0.82, edgecolor="white", linewidth=0.25)
    ax.axvline(0, color=NAVY, lw=0.9)
    ax.axvline(np.mean(paired), color=ORANGE, lw=1.3, ls="--")
    ax.set_xlabel("Per-base-circuit MAE difference vs unmitigated")
    ax.set_ylabel("Base circuits")
    ax.set_title("Average gain coexists with circuit-level reversals")
    ax.text(0.04, 0.92, f"mean {np.mean(paired):+.4f}", transform=ax.transAxes, color=ORANGE)

    ax = axes[0, 1]
    panel(ax, "b")
    metrics = ["mae", "p95_absolute_error", "worst_family_mae", "worst_noise_family_mae", "worst_strength_mae", "worst_shot_budget_mae"]
    labels = ["Mean", "95th pct.", "Worst family", "Worst noise", "Worst strength", "Worst shots"]
    ratios = [within["raw_metrics"]["safe_simplex"][metric] / within["raw_metrics"]["noisy_sampled"][metric] for metric in metrics]
    y = np.arange(len(metrics))
    ax.barh(y, ratios, color=[TEAL if value < 1 else ORANGE for value in ratios])
    ax.axvline(1, color=NAVY, lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Error ratio (global anchor / unmitigated)")
    ax.set_title("Mean, tail and prespecified worst groups")

    ax = axes[1, 0]
    panel(ax, "c")
    methods = ["rf", "simplex", "safe_simplex", "safe_ridge", "safe_cell_simplex"]
    labels = ["RF", "4-model\nsimplex", "Global\nanchor", "Anchored\nridge", "Tier-cell\ndiagnostic"]
    failure = [100 * within["raw_metrics"][method]["failure_rate"] for method in methods]
    ax.bar(np.arange(len(methods)), failure, color=[BLUE, GOLD, "#62B6A7", "#4C9288", TEAL])
    ax.set_xticks(np.arange(len(methods)), labels)
    ax.set_ylabel("Point predictions worse than unmitigated (%)")
    ax.set_title("The anchor reduces, but does not eliminate, failures")

    ax = axes[1, 1]
    panel(ax, "d")
    methods = ["noisy_sampled", "rf", "simplex", "safe_simplex", "safe_cell_simplex"]
    labels = ["Unmitigated", "RF", "Simplex", "Global anchor", "Tier-cell diagnostic"]
    colors = [GREY, BLUE, GOLD, "#62B6A7", TEAL]
    for method, label, color in zip(methods, labels, colors):
        metric = within["raw_metrics"][method]
        ax.scatter(metric["interval_mean_width"], metric["interval_condition_circuit_joint_coverage"], s=36, color=color, label=label)
    ax.axhline(0.90, color=NAVY, lw=0.8, ls="--")
    ax.set_xlabel("Mean interval width")
    ax.set_ylabel("Condition–circuit joint coverage")
    ax.set_ylim(0.88, 0.95)
    ax.set_title("Grouped conformal intervals remain calibrated")
    ax.legend(frameon=False, ncol=2)
    save(fig, "fig4_reliability")


def figure5(summary: dict[str, object]) -> None:
    order = [
        ("circuit_family", "hardware_efficient", "Hardware-efficient family"),
        ("circuit_family", "ising_trotter", "Ising family"),
        ("circuit_family", "random_clifford", "Random Clifford family"),
        ("circuit_family", "warm_start_qaoa", "Nonuniform QAOA-like family"),
        ("deep_circuits", "layers_7_8", "Deep circuits (layers 7–8)"),
        ("noise_family", "coherent_overrotation", "Coherent Z/ZZ phase error"),
        ("noise_family", "damping_dephasing", "Damping/dephasing"),
        ("noise_family", "depolarizing_readout", "Depolarizing/readout"),
        ("noise_strength", "2", "Strongest noise"),
        ("shot_budget", "128", "128 shots"),
        ("observable", "two_qubit_ZZ", "Unseen two-qubit ZZ"),
    ]
    records = {(row["scenario"], row["label"]): row for row in summary["headline"]["shifts"]}
    rows = [records[(scenario, label)] for scenario, label, _ in order]
    labels = [display for _, _, display in order]
    effect = np.asarray([row["paired_difference"] for row in rows])
    low = np.asarray([row["ci_low"] for row in rows])
    high = np.asarray([row["ci_high"] for row in rows])
    fig = plt.figure(figsize=(7.2, 6.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.55, 1))
    ax = fig.add_subplot(grid[0, :])
    panel(ax, "a")
    y = np.arange(len(rows))
    colors = [ORANGE if value > 0 else TEAL for value in effect]
    ax.errorbar(effect, y, xerr=[effect - low, high - effect], fmt="none", ecolor=GREY, elinewidth=1, capsize=2)
    ax.scatter(effect, y, c=colors, s=28, zorder=3)
    ax.axvline(0, color=NAVY, lw=0.9)
    ax.axhspan(-0.5, 3.5, color="#FEF0EA", alpha=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired MAE difference vs unmitigated estimate")
    ax.set_title("Transfer depends on what moved")
    ax.text(0.995, 0.02, "worse →", ha="right", va="bottom", transform=ax.transAxes, color=ORANGE)
    ax.text(0.005, 0.02, "← better", ha="left", va="bottom", transform=ax.transAxes, color=TEAL)

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "b")
    coverage = 100 * np.asarray([row["coverage90"] for row in rows])
    ax.barh(y, coverage, color=[ORANGE if value < 88 else TEAL for value in coverage])
    ax.axvline(90, color=NAVY, lw=0.8, ls="--")
    short_labels = [
        "Hardware-efficient",
        "Ising",
        "Random Clifford",
        "Nonuniform QAOA-like",
        "Deep",
        "Coherent Z/ZZ",
        "Damping",
        "Depol./readout",
        "Strongest noise",
        "128 shots",
        "Unseen ZZ",
    ]
    ax.set_yticks(y, short_labels, fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("Joint coverage (%)")
    ax.set_title("Grouped 90% coverage")

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "c")
    width = np.asarray([row["interval_width"] for row in rows])
    ax.barh(y, width, color=BLUE)
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.set_xlabel("Mean interval width")
    ax.set_title("Shift uncertainty can become impractical")
    save(fig, "fig5_distribution_shifts")


def supplementary(summary: dict[str, object]) -> None:
    curve = load_json(RESULTS / "confirmation/v2/training_size_curve.json")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    methods = [
        "noisy_sampled",
        "rf",
        "simplex",
        "safe_simplex",
        "raw_cell_affine",
        "safe_cell_simplex",
    ]
    labels = [
        "Unmitigated",
        "RF",
        "Simplex",
        "Global anchor",
        "Unmitigated-only cell",
        "Tier-cell diagnostic",
    ]
    colors = [GREY, BLUE, GOLD, "#62B6A7", "#A7A9AC", TEAL]
    for ax, metric, title in zip(
        axes.flat,
        (
            "mae",
            "p95_absolute_error",
            "interval_condition_circuit_joint_coverage",
            "interval_mean_width",
        ),
        ("Mean error", "Tail error", "Joint coverage", "Finite interval width"),
    ):
        for method, label, color in zip(methods, labels, colors):
            x = np.asarray([cell["training_size"] for cell in curve["cells"]])
            y = np.asarray(
                [
                    np.nan if cell["methods"][method][metric] is None else cell["methods"][method][metric]
                    for cell in curve["cells"]
                ],
                dtype=float,
            )
            finite_interval = np.asarray(
                [
                    cell["methods"][method]["finite_interval_available"]
                    for cell in curve["cells"]
                ],
                dtype=bool,
            )
            if metric in {
                "interval_condition_circuit_joint_coverage",
                "interval_mean_width",
            }:
                x, y = x[finite_interval], y[finite_interval]
            ax.plot(x, y, marker="o", ms=3, label=label, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(curve["training_sizes"], [str(value) for value in curve["training_sizes"]])
        ax.set_xlabel("Training-partition base circuits")
        ax.set_title(title)
    for ax in axes[1]:
        ax.text(
            0.02,
            0.04,
            "n=32: no finite nominal 90% radius",
            transform=ax.transAxes,
            fontsize=6.4,
            color=GREY,
        )
    axes[0, 0].legend(frameon=False, ncol=2)
    save(fig, "figs1_training_size_all_metrics")

    sensitivity = summary["headline"]["within_domain"]["failure_sensitivity"]
    tier = summary["headline"]["within_domain"]["tier_metadata_sensitivity"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": (1.05, 1.25)},
        constrained_layout=True,
    )
    ax = axes[0]
    panel(ax, "a")
    methods = ["rf", "simplex", "safe_simplex", "raw_cell_affine", "safe_cell_simplex"]
    method_labels = [
        "RF",
        "4-model\nsimplex",
        "Global\nanchor",
        "Matched\ncontrol",
        "Tier-cell\ndiagnostic",
    ]
    x = np.arange(len(methods))
    strict = 100 * np.asarray([sensitivity[method]["strict_zero_margin"] for method in methods])
    declared = 100 * np.asarray([sensitivity[method]["declared_margin"] for method in methods])
    ax.bar(x - 0.18, strict, width=0.36, color=GREY, label="Strict zero margin")
    ax.bar(x + 0.18, declared, width=0.36, color=TEAL, label="Declared 10⁻⁶ margin")
    ax.set_xticks(x, method_labels)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_ylabel("Failure rate (%)")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fontsize=6.2,
    )
    ax.set_title("Numerical-tie sensitivity")
    ax = axes[1]
    panel(ax, "b")
    matrix = np.asarray(
        [
            [tier["mae_by_true_and_assumed_tier"][str(true)][str(assumed)] for assumed in range(3)]
            for true in range(3)
        ]
    )
    image = ax.imshow(matrix, cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(range(3), ["low", "medium", "high"])
    ax.set_yticks(range(3), ["low", "medium", "high"])
    ax.set_xlabel("Assumed tier")
    ax.set_ylabel("Simulator-known true tier")
    ax.set_title(
        "Tier-label sensitivity\n"
        f"unknown → global {tier['unknown_tier_fallback_mae']:.3f}; "
        f"unmitigated-only cell {tier['raw_only_matched_control_mae']:.3f}",
        fontsize=8,
    )
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, label="MAE")
    save(fig, "figs2_failure_margin_sensitivity")

    archived = load_json(RESULTS / "development/archived_ensemble_cv.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    datasets = list(archived["datasets"])
    for ax, metric, title in zip(axes, ("mae", "p95_absolute_error"), ("Mean error", "Tail error")):
        for method, label, color in (("rf", "RF", BLUE), ("simplex", "Simplex", GOLD), ("ridge", "Ridge", TEAL)):
            values = [archived["datasets"][dataset]["variants"]["raw"]["metrics"][method][metric] for dataset in datasets]
            ax.plot(range(len(datasets)), values, marker="o", ms=3.5, label=label, color=color)
        ax.set_xticks(range(len(datasets)), [name.replace("ising_", "").replace("_", "\n") for name in datasets])
        ax.set_title(title)
    axes[0].legend(frameon=False, ncol=3)
    save(fig, "figs3_archived_ensemble_tradeoffs")


def main() -> None:
    style()
    summary = load_json(PUBLIC_OUTPUT / "results/frozen/frozen_summary.json")
    figure1(summary)
    figure2(summary)
    figure3(summary)
    figure4(summary)
    figure5(summary)
    supplementary(summary)
    print("generated 8 figure assets as transparent PDF and PNG")


if __name__ == "__main__":
    main()
