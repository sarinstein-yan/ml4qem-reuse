#!/usr/bin/env python3
"""Assemble the public summary and tables from frozen analysis inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ml4qem_reuse.ensemble import benjamini_hochberg
from ml4qem_reuse.serialization import strict_json_dumps
from ml4qem_reuse.workflows._paths import output_root, workspace_root


PROJECT = workspace_root()
RESULTS = PROJECT / "results"
PUBLIC_OUTPUT = output_root()
GENERATED = PUBLIC_OUTPUT / "tables" / "latex"
FROZEN = PUBLIC_OUTPUT / "results" / "frozen"


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_gnu_time(path: Path) -> dict[str, float]:
    """Parse the stable fields used from a GNU time -v record."""

    values = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.strip().split(": ", 1)
        values[key] = value
    elapsed_text = values["Elapsed (wall clock) time (h:mm:ss or m:ss)"]
    parts = [float(item) for item in elapsed_text.split(":")]
    elapsed = 0.0
    for item in parts:
        elapsed = 60.0 * elapsed + item
    return {
        "wall_seconds": elapsed,
        "user_seconds_including_children": float(values["User time (seconds)"]),
        "system_seconds_including_children": float(values["System time (seconds)"]),
        "cpu_percent": float(values["Percent of CPU this job got"].rstrip("%")),
        "gnu_time_max_rss_kib": float(values["Maximum resident set size (kbytes)"]),
    }


def format_effect(effect: dict[str, float]) -> str:
    return f"{effect['estimate']:+.4f} [{effect['ci_low']:+.4f}, {effect['ci_high']:+.4f}]"


def parse_cell_key(key: str) -> tuple[int, int]:
    strength, shots = key.removeprefix("S").split(":N")
    return int(strength), int(shots)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise adjusted P values, valid under arbitrary dependence."""

    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate(
        (len(ranked) - np.arange(len(ranked))) * ranked
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def main() -> None:
    paths = {
        "audit": RESULTS / "development/published_result_audit.json",
        "equivalence": RESULTS / "development/feature_equivalence_atol5e-7.json",
        "demo1_portability": RESULTS / "confirmation/demo1_rf_portability.json",
        "demo2_portability": RESULTS / "confirmation/demo2_rf_portability.json",
        "archived_hardware": RESULTS / "confirmation/archived_hardware_reanalysis.json",
        "training_curve": RESULTS / "confirmation/v2/training_size_curve.json",
        "within": RESULTS / "confirmation/v2/local_within_n512_folds0-4_seed0.json",
        "within_predictions": RESULTS
        / "confirmation/v2/local_within_n512_folds0-4_seed0_predictions.npz",
        "model_seeds": RESULTS / "confirmation/v2/model_seed_sensitivity.json",
        "shifts": RESULTS / "confirmation/v2/local_shifts_aggregated_seed0.json",
        "observable": RESULTS / "confirmation/v2/observable_shift_seed0.json",
        "observable_predictions": RESULTS
        / "confirmation/v2/observable_shift_seed0_predictions.npz",
        "stage_audit": RESULTS / "confirmation/v2/stage_independence_audit.json",
        "archived_ensembles": RESULTS / "development/archived_ensemble_cv.json",
        "benchmark_manifest": PROJECT
        / "data/derived/local_benchmark_confirmation_v2_manifest.json",
        "resource_fit": RESULTS
        / "confirmation/v2/local_within_n512_folds0-4_seed0.json",
        "resource_fit_time": RESULTS
        / "resources/v2/local_within_n512_folds0-4_seed0_time.txt",
        "resource_shifts_time": RESULTS / "resources/v2/local_shifts_seed0_time.txt",
        "resource_observable_time": RESULTS
        / "resources/v2/observable_shift_seed0_time.txt",
        "model_storage": RESULTS / "resources/v2/model_storage_seed0.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    data = {key: read(path) for key, path in paths.items() if path.suffix == ".json"}

    within = data["within"]
    failure_margin = within["failure_definition"]["absolute_error_margin"]
    with np.load(paths["within_predictions"], allow_pickle=False) as archive:
        target = archive["target"]
        noisy = archive["raw__noisy_sampled__prediction"]
        failure_sensitivity = {}
        for method in (
            "rf",
            "simplex",
            "safe_simplex",
            "safe_ridge",
            "raw_cell_affine",
            "safe_cell_simplex",
        ):
            prediction = archive[f"raw__{method}__prediction"]
            candidate_error = np.abs(prediction - target)
            reference_error = np.abs(noisy - target)
            failure_sensitivity[method] = {
                "strict_zero_margin": float(np.mean(candidate_error > reference_error)),
                "declared_margin": float(
                    np.mean(candidate_error > reference_error + failure_margin)
                ),
            }

    shift_rows = []
    for scenario, labels in data["shifts"]["scenarios"].items():
        for label, cell in labels.items():
            # The global anchor is the operational comparison because it does
            # not require a calibrated noise-strength tier at deployment.  The
            # cell-conditioned variant is retained as a metadata diagnostic.
            method = "safe_simplex"
            metrics = cell["metrics"][method]
            noisy_metrics = cell["metrics"]["noisy_sampled"]
            effect = cell["paired_circuit_mae_effects"]["noisy_sampled"][method]
            shift_rows.append(
                {
                    "scenario": scenario,
                    "label": label,
                    "method": method,
                    "base_circuits": cell["base_circuits"],
                    "rows": cell["rows"],
                    "noisy_mae": noisy_metrics["mae"],
                    "candidate_mae": metrics["mae"],
                    "paired_difference": effect["estimate"],
                    "ci_low": effect["ci_low"],
                    "ci_high": effect["ci_high"],
                    "p_value": effect["signflip_p_value_two_sided"],
                    "signflip_draws": effect["draws"],
                    "p_value_at_sampling_floor": bool(
                        np.isclose(
                            effect["signflip_p_value_two_sided"],
                            1.0 / (effect["draws"] + 1),
                        )
                    ),
                    "failure_rate": metrics["failure_rate"],
                    "coverage90": metrics["interval_condition_circuit_joint_coverage"],
                    "interval_width": metrics["interval_mean_width"],
                }
            )
    observable = data["observable"]
    observable_method = "safe_simplex"
    with np.load(paths["observable_predictions"], allow_pickle=False) as archive:
        observable_rows = int(len(archive["target"]))
        observable_base_circuits = int(len(np.unique(archive["base_index"])))
    observable_effect = observable["paired_circuit_mae_effects"]["noisy_sampled"][
        observable_method
    ]
    shift_rows.append(
        {
            "scenario": "observable",
            "label": "two_qubit_ZZ",
            "method": observable_method,
            "base_circuits": observable_base_circuits,
            "rows": observable_rows,
            "noisy_mae": observable["raw_metrics"]["noisy_sampled"]["mae"],
            "candidate_mae": observable["raw_metrics"][observable_method]["mae"],
            "paired_difference": observable_effect["estimate"],
            "ci_low": observable_effect["ci_low"],
            "ci_high": observable_effect["ci_high"],
            "p_value": observable_effect["signflip_p_value_two_sided"],
            "signflip_draws": observable_effect["draws"],
            "p_value_at_sampling_floor": bool(
                np.isclose(
                    observable_effect["signflip_p_value_two_sided"],
                    1.0 / (observable_effect["draws"] + 1),
                )
            ),
            "failure_rate": observable["raw_metrics"][observable_method]["failure_rate"],
            "coverage90": observable["raw_metrics"][observable_method][
                "interval_base_circuit_joint_coverage"
            ],
            "interval_width": observable["raw_metrics"][observable_method][
                "interval_mean_width"
            ],
        }
    )
    q_values = benjamini_hochberg(
        np.asarray([row["p_value"] for row in shift_rows], dtype=float)
    )
    holm_values = holm_adjust(
        np.asarray([row["p_value"] for row in shift_rows], dtype=float)
    )
    for row, q_value, holm_value in zip(shift_rows, q_values, holm_values):
        row["q_value_bh_11_tests"] = float(q_value)
        row["p_value_holm_11_tests"] = float(holm_value)

    training_rows = []
    for cell in data["training_curve"]["cells"]:
        for method, metrics in cell["methods"].items():
            effect = cell["paired_mae_effect_vs_noisy"].get(method)
            training_rows.append(
                {
                    "training_size": cell["training_size"],
                    "method": method,
                    **metrics,
                    "paired_difference": "" if effect is None else effect["estimate"],
                    "ci_low": "" if effect is None else effect["ci_low"],
                    "ci_high": "" if effect is None else effect["ci_high"],
                }
            )

    cell_weight_rows = []
    for record in within["fold_records"]:
        for key, weights in sorted(
            record["safe_cell_simplex_weights"].items(), key=lambda item: parse_cell_key(item[0])
        ):
            strength, shots = parse_cell_key(key)
            cell_weight_rows.append(
                {
                    "fold": record["fold"],
                    "strength_tier": strength,
                    "shots": shots,
                    **{
                        name: value
                        for name, value in zip(
                            within["safety_ensemble_base_order"], weights, strict=True
                        )
                    },
                }
            )

    selected_methods = (
        "noisy_sampled",
        "rf",
        "simplex",
        "safe_simplex",
        "safe_ridge",
        "raw_cell_affine",
        "safe_cell_simplex",
    )
    within_rows = []
    for method in selected_methods:
        raw = within["raw_metrics"][method]
        clipped = within["clipped_metrics"][method]
        within_rows.append(
            {
                "method": method,
                **{f"raw_{key}": value for key, value in raw.items()},
                **{f"clipped_{key}": value for key, value in clipped.items()},
            }
        )

    seed_rows = []
    for method in ("rf", "safe_simplex", "safe_ridge", "safe_cell_simplex"):
        for metric in ("mae", "failure_rate", "p95_absolute_error"):
            seed_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    **data["model_seeds"]["metrics"]["raw_metrics"][method][metric],
                }
            )

    archived_ensemble_rows = []
    for dataset, record in data["archived_ensembles"]["datasets"].items():
        for method in ("rf", "mean", "median", "simplex", "ridge"):
            metrics = record["variants"]["raw"]["metrics"][method]
            archived_ensemble_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "p95_absolute_error": metrics["p95_absolute_error"],
                    "failure_rate": metrics["failure_rate"],
                    "physical_range_violation_rate": metrics[
                        "physical_range_violation_rate"
                    ],
                }
            )

    resource_fit_time = parse_gnu_time(paths["resource_fit_time"])
    resource_shifts_time = parse_gnu_time(paths["resource_shifts_time"])
    resource_observable_time = parse_gnu_time(paths["resource_observable_time"])
    model_storage = data["model_storage"]
    training_time_paths = sorted((RESULTS / "resources/v2").glob("local_within_n*_time.txt"))
    training_wall_times = [parse_gnu_time(path)["wall_seconds"] for path in training_time_paths]
    benchmark = data["benchmark_manifest"]
    hardware = data["archived_hardware"]
    # The large DOI download is deliberately not duplicated in the compact
    # release data. Its byte count is provenance metadata, not a runtime
    # dependency of summary regeneration.
    doi_archive_bytes = 659_720_205
    resource_rows = [
        {
            "evidence": "DOI artifact recovery",
            "data_or_workload": f"{doi_archive_bytes / 1e6:.1f} MB DOI archive; 9 tests; two demos",
            "classical_cost": "3.17 s cold tests; 2.50 s after one-line repair",
            "qpu_equivalent": "archived data only",
            "new_qpu_jobs": 0,
        },
        {
            "evidence": "Legacy-to-current portability",
            "data_or_workload": "80 circuits; two 30-seed RF repeats",
            "classical_cost": "CPU only; no retraining discrepancy beyond roundoff",
            "qpu_equivalent": "none",
            "new_qpu_jobs": 0,
        },
        {
            "evidence": "Local benchmark generation",
            "data_or_workload": f"{benchmark['base_circuits']} base circuits; {benchmark['sampled_rows']:,} rows",
            "classical_cost": (
                f"{benchmark['resources']['wall_seconds']:.1f} s wall; "
                f"{benchmark['resources']['peak_rss_kib'] / 1024:.0f} MiB main-process peak; "
                f"{benchmark['resources']['threads_requested']} requested threads"
            ),
            "qpu_equivalent": "328.9 M nominal shots; local simulation only",
            "new_qpu_jobs": 0,
        },
        {
            "evidence": "In-domain grouped confirmation",
            "data_or_workload": "five confirmatory folds; released-style per-observable RF",
            "classical_cost": (
                f"{resource_fit_time['wall_seconds']:.1f} s wall; "
                f"{resource_fit_time['user_seconds_including_children']:.0f} s user; "
                f"{resource_fit_time['cpu_percent']:.0f}% CPU; "
                f"{resource_fit_time['gnu_time_max_rss_kib'] / 1024:.0f} MiB GNU-time max RSS"
                f"; {model_storage['serialized_decimal_megabytes']:.1f} MB measured "
                "serialized model footprint (models omitted)"
            ),
            "qpu_equivalent": "no additional shots beyond local data",
            "new_qpu_jobs": 0,
        },
        {
            "evidence": "Training-size and seed sensitivities",
            "data_or_workload": (
                "5 nested availability-pool sizes; seeds 0--4 at n=512; "
                "9 unique fits"
            ),
            "classical_cost": (
                f"{min(training_wall_times):.1f}--{max(training_wall_times):.1f} s wall per fit; "
                "CPU only"
            ),
            "qpu_equivalent": "no additional shots beyond local data",
            "new_qpu_jobs": 0,
        },
        {
            "evidence": "One-axis robustness suite",
            "data_or_workload": "10 circuit/depth/noise/shot holdouts; 11th test is observable shift",
            "classical_cost": (
                f"{resource_shifts_time['wall_seconds']:.1f} s wall; "
                f"{resource_shifts_time['user_seconds_including_children']:.0f} s user; "
                f"{resource_shifts_time['cpu_percent']:.0f}% CPU; "
                f"{resource_shifts_time['gnu_time_max_rss_kib'] / 1024:.0f} MiB GNU-time max RSS"
            ),
            "qpu_equivalent": "no additional shots beyond local data",
            "new_qpu_jobs": 0,
        },
        {
            "evidence": "Unseen-ZZ transfer",
            "data_or_workload": "train on Z0/Z1; test on Z0Z1/Z2Z3; five folds",
            "classical_cost": (
                f"{resource_observable_time['wall_seconds']:.1f} s wall; "
                f"{resource_observable_time['user_seconds_including_children']:.0f} s user; "
                f"{resource_observable_time['cpu_percent']:.0f}% CPU; "
                f"{resource_observable_time['gnu_time_max_rss_kib'] / 1024:.0f} MiB GNU-time max RSS"
            ),
            "qpu_equivalent": "no additional shots beyond local data",
            "new_qpu_jobs": 0,
        },
        {
            "evidence": "Archived ibm_algiers reanalysis",
            "data_or_workload": "500 training + 2,500 test circuits",
            "classical_cost": "statistics on distributed predictions",
            "qpu_equivalent": "30 M ML4QEM versus 50 M ZNE source shots",
            "new_qpu_jobs": 0,
        },
    ]

    headline = {
        "benchmark_identity": {
            "dataset_family_id": benchmark["dataset_family_id"],
            "dataset_instance_id": benchmark["dataset_instance_id"],
        },
        "model_storage": model_storage,
        "source_recovery": {
            "figure2_rows": data["audit"]["figure2"]["distributed_result_rows"],
            "figure2_missing_rows": data["audit"]["figure2"]["missing_relative_to_caption"],
            "figure2_max_abs_difference": data["audit"]["figure2"]["max_absolute_difference"],
            "figure3_cells": data["audit"]["figure3"]["cells_compared"],
            "figure3_max_abs_difference": data["audit"]["figure3"]["max_absolute_difference"],
            "figure4_cells": data["audit"]["figure4"]["cells_compared"],
            "figure4_max_abs_difference": data["audit"]["figure4"]["max_absolute_difference"],
        },
        "portability": {
            "feature_cases": data["equivalence"]["n_cases"],
            "feature_max_abs_difference": data["equivalence"]["global_max_absolute_difference"],
            "demo1_prediction_max_abs_difference": data["demo1_portability"][
                "global_prediction_max_absolute_difference"
            ],
            "demo2_prediction_max_abs_difference": data["demo2_portability"][
                "global_prediction_max_absolute_difference"
            ],
        },
        "within_domain": {
            "noisy": within["raw_metrics"]["noisy_sampled"],
            "rf": within["raw_metrics"]["rf"],
            "safe_simplex": within["raw_metrics"]["safe_simplex"],
            "raw_cell_affine": within["raw_metrics"]["raw_cell_affine"],
            "safe_cell_simplex_diagnostic": within["raw_metrics"]["safe_cell_simplex"],
            "paired_effect_vs_noisy": within["raw_paired_circuit_mae_effects"][
                "noisy_sampled"
            ]["safe_simplex"],
            "cell_diagnostic_effect_vs_noisy": within[
                "raw_paired_circuit_mae_effects"
            ]["noisy_sampled"]["safe_cell_simplex"],
            "tier_metadata_sensitivity": within["tier_metadata_sensitivity"],
            "fold_effects": within["fold_effects"],
            "failure_sensitivity": failure_sensitivity,
            "stage_independence_audit": data["stage_audit"],
        },
        "archived_hardware": {
            "metrics": hardware["metrics"],
            "effects": hardware["paired_circuit_mae_effects"],
            "resources": hardware["resource_accounting"],
        },
        "shifts": shift_rows,
        "observable_transfer_exploratory_tier_cell": {
            "metrics": observable["raw_metrics"]["safe_cell_simplex"],
            "effect_vs_unmitigated": observable["paired_circuit_mae_effects"][
                "noisy_sampled"
            ]["safe_cell_simplex"],
            "multiplicity_status": (
                "exploratory metadata-conditioned diagnostic; not included in the "
                "prespecified 11-test multiplicity family"
            ),
        },
    }
    summary = {
        "schema_version": 1,
        "evidence_freeze": "2026-08-15",
        "headline": headline,
        "failure_definition": within["failure_definition"],
        "resource_rows": resource_rows,
        "input_files": {
            key: {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "hardware_jobs_submitted": 0,
    }

    FROZEN.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    (FROZEN / "frozen_summary.json").write_text(
        strict_json_dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(FROZEN / "table1_resources.csv", resource_rows)
    write_csv(FROZEN / "table_s1_training_size.csv", training_rows)
    write_csv(FROZEN / "table_s2_within_domain.csv", within_rows)
    write_csv(FROZEN / "table_s3_distribution_shifts.csv", shift_rows)
    write_csv(FROZEN / "table_s4_model_seed_sensitivity.csv", seed_rows)
    write_csv(FROZEN / "table_s5_archived_ensembles.csv", archived_ensemble_rows)
    write_csv(FROZEN / "table_s6_cell_weights.csv", cell_weight_rows)

    overlap_effect = headline["within_domain"]["stage_independence_audit"][
        "confirmation_exclusion_sensitivity"
    ]["paired_global_anchor_mae_effect_vs_unmitigated"]
    observable_cell_effect = headline["observable_transfer_exploratory_tier_cell"][
        "effect_vs_unmitigated"
    ]
    observable_shift_row = next(row for row in shift_rows if row["scenario"] == "observable")
    shift_floor_holm = min(
        row["p_value_holm_11_tests"]
        for row in shift_rows
        if row["p_value_at_sampling_floor"]
    )
    macros = {
        "FeatureCases": f"{headline['portability']['feature_cases']}",
        "FeatureMaxDiff": f"{headline['portability']['feature_max_abs_difference']:.3g}",
        "DemoOneMaxDiff": f"{headline['portability']['demo1_prediction_max_abs_difference']:.3g}",
        "DemoTwoMaxDiff": f"{headline['portability']['demo2_prediction_max_abs_difference']:.3g}",
        "WithinNoisyMAE": f"{headline['within_domain']['noisy']['mae']:.4f}",
        "WithinRFMAE": f"{headline['within_domain']['rf']['mae']:.4f}",
        "WithinSafeMAE": f"{headline['within_domain']['safe_simplex']['mae']:.4f}",
        "WithinSafeEffect": f"{headline['within_domain']['paired_effect_vs_noisy']['estimate']:.4f}",
        "WithinSafeEffectLow": f"{headline['within_domain']['paired_effect_vs_noisy']['ci_low']:.4f}",
        "WithinSafeEffectHigh": f"{headline['within_domain']['paired_effect_vs_noisy']['ci_high']:.4f}",
        "WithinSafeFailurePct": f"{100 * headline['within_domain']['safe_simplex']['failure_rate']:.1f}",
        "WithinCellMAE": f"{headline['within_domain']['safe_cell_simplex_diagnostic']['mae']:.4f}",
        "WithinCellFailurePct": f"{100 * headline['within_domain']['safe_cell_simplex_diagnostic']['failure_rate']:.1f}",
        "WithinCellEffect": f"{headline['within_domain']['cell_diagnostic_effect_vs_noisy']['estimate']:.4f}",
        "WithinCellEffectLow": f"{headline['within_domain']['cell_diagnostic_effect_vs_noisy']['ci_low']:.4f}",
        "WithinCellEffectHigh": f"{headline['within_domain']['cell_diagnostic_effect_vs_noisy']['ci_high']:.4f}",
        "WithinRawCellMAE": f"{headline['within_domain']['raw_cell_affine']['mae']:.4f}",
        "OverlapExcludedEffect": f"{overlap_effect['estimate']:.4f}",
        "ObservableCellEffect": f"{observable_cell_effect['estimate']:.4f}",
        "ObservableCellEffectLow": f"{observable_cell_effect['ci_low']:.4f}",
        "ObservableCellEffectHigh": f"{observable_cell_effect['ci_high']:.4f}",
        "ObservableHolmP": f"{observable_shift_row['p_value_holm_11_tests']:.3f}",
        "ShiftFloorHolmP": f"{shift_floor_holm:.4f}",
        "ArchivedNoisyMAE": f"{hardware['metrics']['noisy']['mae']:.4f}",
        "ArchivedZNEMAE": f"{hardware['metrics']['zne']['mae']:.4f}",
        "ArchivedRFMAE": f"{hardware['metrics']['rf']['mae']:.4f}",
        "ArchivedRFEffect": f"{hardware['paired_circuit_mae_effects']['noisy']['rf']['estimate']:.4f}",
        "ArchivedRFEffectLow": f"{hardware['paired_circuit_mae_effects']['noisy']['rf']['ci_low']:.4f}",
        "ArchivedRFEffectHigh": f"{hardware['paired_circuit_mae_effects']['noisy']['rf']['ci_high']:.4f}",
    }
    (GENERATED / "metrics.tex").write_text(
        "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in macros.items())
    )

    table_lines = [
        r"\begin{tabularx}{\textwidth}{@{}p{0.19\textwidth}X X p{0.18\textwidth}c@{}}",
        r"\toprule",
        r"Evidence stream & Data or workload & Classical cost & QPU-equivalent accounting & New QPU jobs \\",
        r"\midrule",
    ]
    for row in resource_rows:
        table_lines.append(
            " & ".join(
                tex_escape(str(row[key]))
                for key in (
                    "evidence",
                    "data_or_workload",
                    "classical_cost",
                    "qpu_equivalent",
                    "new_qpu_jobs",
                )
            )
            + r" \\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabularx}"])
    (GENERATED / "table1.tex").write_text("\n".join(table_lines) + "\n")

    training_tex = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Training circuits & Unmitigated MAE & Global-anchor MAE & Paired difference & 95\% interval \\",
        r"\midrule",
    ]
    for cell in data["training_curve"]["cells"]:
        effect = cell["paired_mae_effect_vs_noisy"]["safe_simplex"]
        training_tex.append(
            f"{cell['training_size']} & {cell['methods']['noisy_sampled']['mae']:.6f} & "
            f"{cell['methods']['safe_simplex']['mae']:.6f} & {effect['estimate']:+.6f} & "
            f"[{effect['ci_low']:+.6f}, {effect['ci_high']:+.6f}] \\\\"
        )
    training_tex.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "table_s_training.tex").write_text("\n".join(training_tex) + "\n")

    within_tex = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & MAE & RMSE & P95 & Failure & Worst family & Joint coverage & Width \\",
        r"\midrule",
    ]
    method_labels = {
        "noisy_sampled": "Unmitigated",
        "rf": "RF",
        "simplex": "Four-model simplex",
        "safe_simplex": "Global anchor",
        "safe_ridge": "Unmitigated-anchored ridge",
        "raw_cell_affine": "Unmitigated-only cell affine",
        "safe_cell_simplex": "Tier-cell diagnostic",
    }
    for method in selected_methods:
        metric = within["raw_metrics"][method]
        within_tex.append(
            f"{method_labels[method]} & {metric['mae']:.5f} & {metric['rmse']:.5f} & "
            f"{metric['p95_absolute_error']:.5f} & {100 * metric['failure_rate']:.1f}\\% & "
            f"{metric['worst_family_mae']:.5f} & "
            f"{100 * metric['interval_condition_circuit_joint_coverage']:.1f}\\% & "
            f"{metric['interval_mean_width']:.3f} \\\\"
        )
    within_tex.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "table_s_within.tex").write_text("\n".join(within_tex) + "\n")

    cell_weights_tex = [
        r"\begin{longtable}{rrrrrrrr}",
        r"\caption{Fold- and cell-specific metadata-conditioned simplex weights. Columns follow the five-expert implementation order. Values are shown to six decimals; the machine-readable CSV retains full precision.}\label{tab:s-cell-weights}\\",
        r"\toprule",
        r"Fold & Tier & Shots & Raw & Linear & RF & HGB & MLP \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Fold & Tier & Shots & Raw & Linear & RF & HGB & MLP \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in cell_weight_rows:
        cell_weights_tex.append(
            f"{row['fold']} & {row['strength_tier']} & {row['shots']} & "
            f"{row['noisy_sampled']:.6f} & {row['linear']:.6f} & {row['rf']:.6f} & "
            f"{row['hgb']:.6f} & {row['mlp']:.6f} \\\\"
        )
    cell_weights_tex.extend([r"\bottomrule", r"\end{longtable}"])
    (GENERATED / "table_s_cell_weights.tex").write_text(
        "\n".join(cell_weights_tex) + "\n"
    )

    scenario_labels = {
        "circuit_family": "Circuit family",
        "deep_circuits": "Depth",
        "noise_family": "Noise family",
        "noise_strength": "Noise strength",
        "shot_budget": "Shot budget",
        "observable": "Observable",
    }
    publication_labels = {
        "hardware_efficient": "hardware-efficient",
        "ising_trotter": "Ising Trotter",
        "random_clifford": "random Clifford",
        "warm_start_qaoa": "nonuniform QAOA-like",
        "layers_7_8": "layers 7--8",
        "coherent_overrotation": "coherent $Z/ZZ$ phase error",
        "damping_dephasing": "damping/dephasing",
        "depolarizing_readout": r"depolarizing/\allowbreak readout",
        "2": "strongest tier",
        "128": "128 shots",
        "two_qubit_ZZ": "unseen two-qubit $ZZ$",
    }
    shift_tex = [
        r"\begin{longtable}{@{}p{0.09\textwidth}p{0.14\textwidth}rrp{0.22\textwidth}rrr@{}}",
        r"\caption{Complete prespecified one-axis distribution-shift results for the global tier-agnostic interface. Each holdout cell gives the number $n$ of base-circuit grouping units; descendant rows are not inferential units. Intervals use 10,000 paired grouped-bootstrap draws and are conditional on the frozen fits and splits. Sign-flip diagnostics use 10,000 draws under sign exchangeability and are Holm-adjusted across 11 tests, which controls family-wise error under arbitrary dependence. Zero sign-flip exceedances yield $P_{\rm Holm}=\ShiftFloorHolmP$ rather than an exact tail probability. Failure and grouped joint coverage are percentages.}\label{tab:s-shifts}\\",
        r"\toprule",
        r"Axis & Holdout ($n$ base circuits) & Raw MAE & Candidate MAE & Paired difference [95\% interval] & Holm $P$ & Failure & Coverage \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Axis & Holdout ($n$ base circuits) & Raw MAE & Candidate MAE & Paired difference [95\% interval] & Holm $P$ & Failure & Coverage \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in shift_rows:
        shift_tex.append(
            f"{scenario_labels[row['scenario']]} & "
            f"{publication_labels.get(str(row['label']), tex_escape(str(row['label'])))} "
            f"($n={row['base_circuits']}$) & "
            f"{row['noisy_mae']:.5f} & {row['candidate_mae']:.5f} & "
            f"{row['paired_difference']:+.5f} [{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] & "
            f"{row['p_value_holm_11_tests']:.4f} & "
            f"{100 * row['failure_rate']:.1f}\\% & {100 * row['coverage90']:.1f}\\% \\\\"
        )
    shift_tex.extend([r"\bottomrule", r"\end{longtable}"])
    (GENERATED / "table_s_shifts.tex").write_text("\n".join(shift_tex) + "\n")

    seed_tex = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Confirmatory fold & Global-anchor effect & Tier-cell effect & Test circuits \\",
        r"\midrule",
    ]
    for row in within["fold_effects"]:
        seed_tex.append(
            f"{row['fold']} & {row['safe_simplex_effect_vs_noisy']:+.6f} & "
            f"{row['safe_cell_effect_vs_noisy']:+.6f} & {row['base_circuits']}" + r" \\"
        )
    seed_tex.extend(
        [
            r"\midrule",
        r"Method & Mean MAE & Between-seed s.d. & Range \\",
        r"\midrule",
        ]
    )
    for method in ("rf", "safe_simplex", "safe_ridge", "safe_cell_simplex"):
        values = data["model_seeds"]["metrics"]["raw_metrics"][method]["mae"]
        seed_tex.append(
            f"{method_labels[method]} & {values['mean']:.6f} & {values['sample_sd']:.6f} & "
            f"{values['minimum']:.6f}--{values['maximum']:.6f} \\\\"
        )
    seed_tex.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "table_s_seeds.tex").write_text("\n".join(seed_tex) + "\n")

    archived_tex = [
        r"\begin{longtable}{p{0.22\textwidth}lrrrr}",
        r"\caption{Cross-fitted combinations of the four distributed model predictions. Each Ising dataset contains $n=9{,}000$ source rows and the random-circuit dataset contains $n=1{,}996$. Rows are treated as distinct circuits because no stable cross-depth identifier is available. Failure uses the declared numerical margin.}\label{tab:s-archived-ensemble}\\",
        r"\toprule",
        r"Distributed dataset & Method & MAE & RMSE & P95 & Failure \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Distributed dataset & Method & MAE & RMSE & P95 & Failure \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in archived_ensemble_rows:
        archived_tex.append(
            f"{tex_escape(str(row['dataset']))} & {tex_escape(str(row['method']))} & "
            f"{row['mae']:.5f} & {row['rmse']:.5f} & {row['p95_absolute_error']:.5f} & "
            f"{100 * row['failure_rate']:.1f}\\% \\\\"
        )
    archived_tex.extend([r"\bottomrule", r"\end{longtable}"])
    (GENERATED / "table_s_archived_ensembles.tex").write_text(
        "\n".join(archived_tex) + "\n"
    )

    print(strict_json_dumps({"headline": headline, "outputs": 13}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
