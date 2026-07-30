"""
Phase 13 (partial) - Validation Metrics Comparison: PatchTST vs Mamba

Compares the 4 extended validation statistics (Pearson r, Scatter Index,
Bias, RMS Error) between PatchTST and Mamba, per variable (swh/mwp/mwd),
across all four oceans.

PatchTST numbers are loaded live from each ocean's validation_metrics.json
(Phase 12 output). Mamba numbers are hardcoded below, transcribed from the
teammate's SECOND (re-trained, more accurate) result set - these supersede
an earlier, now-outdated transcription used in the first version of this
script.

Winner is determined by RMS Error (lower = better) - the single unambiguous
accuracy metric of the four. A note flags any row where the RMSE winner and
the correlation-coefficient winner disagree.

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

# Transcribed from the teammate's SECOND (re-trained) Mamba result tables.
# Keys: ocean -> variable -> {r, si, bias, rmse}
MAMBA_METRICS = {
    "atlantic": {
        "swh": {"r": 0.9787, "si": 0.1141, "bias": 0.0607, "rmse": 0.1716},
        "mwp": {"r": 0.9760, "si": 0.0388, "bias": -0.0482, "rmse": 0.3413},
        "mwd": {"r": 0.9279, "si": 0.1018, "bias": -1.3912, "rmse": 18.3805},
    },
    "pacific": {
        "swh": {"r": 0.9811, "si": 0.0690, "bias": 0.0687, "rmse": 0.1539},
        "mwp": {"r": 0.9813, "si": 0.0367, "bias": 0.0761, "rmse": 0.3177},
        "mwd": {"r": 0.8492, "si": 0.4558, "bias": 1.3654, "rmse": 60.2926},
    },
    "bay_of_bengal": {
        "swh": {"r": 0.9750, "si": 0.1043, "bias": 0.0613, "rmse": 0.1833},
        "mwp": {"r": 0.9762, "si": 0.0356, "bias": -0.0150, "rmse": 0.2760},
        "mwd": {"r": 0.8094, "si": 0.4867, "bias": -1.1935, "rmse": 60.0275},
    },
    "arabian_sea": {
        "swh": {"r": 0.9918, "si": 0.0958, "bias": 0.1052, "rmse": 0.1561},
        "mwp": {"r": 0.9810, "si": 0.0330, "bias": -0.0005, "rmse": 0.2667},
        "mwd": {"r": 0.8454, "si": 0.2490, "bias": 1.3285, "rmse": 47.8113},
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
    print("(Mamba numbers: re-trained/updated set)")
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
        "Mamba numbers from the teammate's re-trained models. Computed in real "
        "units across all 20 forecast steps. Winner determined by lower RMSE.\n\n"
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

    print("\n[OK] Validation metrics comparison complete (updated Mamba numbers).")


if __name__ == "__main__":
    main()
