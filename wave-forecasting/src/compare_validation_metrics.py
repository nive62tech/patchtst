"""
Phase 13 (partial) - Validation Metrics Comparison: PatchTST vs Mamba

Compares the 4 extended validation statistics (Pearson r, Scatter Index,
Bias, RMS Error) between PatchTST and Mamba, per variable (swh/mwp/mwd),
across all four oceans.

PatchTST numbers are loaded live from each ocean's validation_metrics.json
(Phase 12 output). Mamba numbers are hardcoded below, transcribed from the
teammate's result tables (no machine-readable Mamba file was available for
this comparison - only screenshotted tables), since these images were
shared directly rather than the Mamba team writing to disk.

Winner is determined by RMS Error (lower = better) - the single unambiguous
accuracy metric of the four (unlike bias, where sign doesn't indicate
better/worse, and unlike correlation/SI, which RMSE already captures the
practical effect of). A note flags any row where the RMSE winner and the
correlation-coefficient winner disagree.

Saves: comparison/patchtst_vs_mamba_validation.csv
       comparison/patchtst_vs_mamba_validation.md
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMPARISON_DIR = ROOT / "comparison"

OCEANS = ["atlantic", "pacific", "bay_of_bengal", "arabian_sea"]
OCEAN_DISPLAY = {
    "atlantic": "Atlantic",
    "pacific": "Pacific",
    "bay_of_bengal": "Bay of Bengal",
    "arabian_sea": "Arabian Sea",
}
VARIABLES = ["swh", "mwp", "mwd"]

# Transcribed directly from the teammate's Mamba result tables (screenshots).
# Keys: ocean -> variable -> {r, si, bias, rmse}
MAMBA_METRICS = {
    "atlantic": {
        "swh": {"r": 0.9748, "si": 0.1025, "bias": 0.0707, "rmse": 0.1800},
        "mwp": {"r": 0.9769, "si": 0.0350, "bias": 0.0049, "rmse": 0.2713},
        "mwd": {"r": 0.8115, "si": 0.4811, "bias": -3.2016, "rmse": 59.4123},
    },
    "pacific": {
        "swh": {"r": 0.9826, "si": 0.0681, "bias": 0.0746, "rmse": 0.1520},
        "mwp": {"r": 0.9823, "si": 0.0346, "bias": 0.0140, "rmse": 0.2994},
        "mwd": {"r": 0.8545, "si": 0.4457, "bias": -0.3156, "rmse": 58.9412},
    },
    "bay_of_bengal": {
        "swh": {"r": 0.9778, "si": 0.1167, "bias": 0.0746, "rmse": 0.1754},
        "mwp": {"r": 0.9753, "si": 0.0390, "bias": -0.0182, "rmse": 0.3432},
        "mwd": {"r": 0.9376, "si": 0.0946, "bias": -0.5789, "rmse": 17.0343},
    },
    "arabian_sea": {
        "swh": {"r": 0.9925, "si": 0.0990, "bias": 0.1126, "rmse": 0.1613},
        "mwp": {"r": 0.9804, "si": 0.0335, "bias": 0.0212, "rmse": 0.2710},
        "mwd": {"r": 0.8657, "si": 0.2307, "bias": -0.5895, "rmse": 44.2710},
    },
}


def load_patchtst_metrics(ocean: str) -> dict:
    path = ROOT / ocean / "results" / "validation_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run {ocean}'s Phase 12 validation_metrics script first.")
    with open(path) as f:
        data = json.load(f)
    return {
        var: {
            "r": data[var]["correlation_coefficient"],
            "si": data[var]["scatter_index"],
            "bias": data[var]["bias"],
            "rmse": data[var]["rmse"],
        }
        for var in VARIABLES
    }


def main():
    print("=" * 70)
    print("PHASE 13 (PARTIAL) - VALIDATION METRICS COMPARISON: PATCHTST vs MAMBA")
    print("=" * 70)

    rows = []
    disagreements = []

    for ocean in OCEANS:
        print(f"\n--- {OCEAN_DISPLAY[ocean]} ---")
        patchtst = load_patchtst_metrics(ocean)
        mamba = MAMBA_METRICS[ocean]

        for var in VARIABLES:
            p = patchtst[var]
            m = mamba[var]

            rmse_winner = "Mamba" if m["rmse"] < p["rmse"] else "PatchTST"
            r_winner = "Mamba" if m["r"] > p["r"] else "PatchTST"
            rmse_pct_diff = (p["rmse"] - m["rmse"]) / m["rmse"] * 100

            if rmse_winner != r_winner:
                disagreements.append(f"{OCEAN_DISPLAY[ocean]} / {var}: RMSE favors {rmse_winner}, r favors {r_winner}")

            for model_name, stats in [("PatchTST", p), ("Mamba", m)]:
                rows.append({
                    "ocean": OCEAN_DISPLAY[ocean],
                    "variable": var,
                    "model": model_name,
                    "correlation_r": stats["r"],
                    "scatter_index": stats["si"],
                    "bias": stats["bias"],
                    "rmse": stats["rmse"],
                    "rmse_winner": rmse_winner,
                    "patchtst_rmse_pct_worse_than_mamba": round(rmse_pct_diff, 1),
                })

            print(
                f"     {var:4s}: PatchTST RMSE={p['rmse']:.4f}  Mamba RMSE={m['rmse']:.4f}  "
                f"-> {rmse_winner} wins (PatchTST is {rmse_pct_diff:+.1f}% vs Mamba)"
            )

    df = pd.DataFrame(rows)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = COMPARISON_DIR / "patchtst_vs_mamba_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # Build markdown summary table (RMSE-based winner, one row per ocean/variable)
    md_rows = []
    for ocean in OCEANS:
        for var in VARIABLES:
            p = load_patchtst_metrics(ocean)[var]
            m = MAMBA_METRICS[ocean][var]
            winner = "Mamba" if m["rmse"] < p["rmse"] else "PatchTST"
            md_rows.append(
                f"| {OCEAN_DISPLAY[ocean]} | {var} | {p['r']:.4f} | {m['r']:.4f} | "
                f"{p['rmse']:.4f} | {m['rmse']:.4f} | **{winner}** |"
            )

    md_content = (
        "# Validation Metrics Comparison — PatchTST vs Mamba\n\n"
        "Computed in real units across all 20 forecast steps. Winner determined "
        "by lower RMSE (the unambiguous accuracy metric of the four).\n\n"
        "| Ocean | Variable | PatchTST r | Mamba r | PatchTST RMSE | Mamba RMSE | Winner (RMSE) |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(md_rows)
        + "\n\nUnits: swh (m), mwp (s), mwd (deg).\n"
    )

    if disagreements:
        md_content += "\n## Rows where RMSE and correlation disagree on the winner\n\n"
        for d in disagreements:
            md_content += f"- {d}\n"

    md_path = COMPARISON_DIR / "patchtst_vs_mamba_validation.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"[SAVED] {md_path}")

    print("\n--- Overall tally (RMSE-based wins, across all 12 ocean/variable rows) ---")
    tally = df[df["model"] == "PatchTST"]["rmse_winner"].value_counts()
    for model, count in tally.items():
        print(f"     {model}: {count}/12")

    if disagreements:
        print(f"\n[INFO] {len(disagreements)} row(s) where RMSE and correlation disagree on winner:")
        for d in disagreements:
            print(f"     - {d}")

    print("\n[OK] Validation metrics comparison complete.")


if __name__ == "__main__":
    main()
