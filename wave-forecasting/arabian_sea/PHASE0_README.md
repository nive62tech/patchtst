# Phase 0 — Folder Skeleton

## What was built
Top-level folder structure for `bay_of_bengal/` and `arabian_sea/`, mirroring
`atlantic/` and `pacific/` exactly: `data/windows/`, `results/`.

## How to run
One-time PowerShell `New-Item` commands, no script.

## Key technical decisions
- Same structure convention as Atlantic/Pacific, per-sea prefix (`bay_of_bengal_`, `arabian_sea_`).
- `data/` and checkpoints/scalers are gitignored (`*/data/`, `*.pt`, `*.pkl`) — only code and `results/` are tracked.

## Result — this run
Folders created successfully for both seas. No files to commit (empty/gitignored dirs).

## Files created/updated
- `bay_of_bengal/data/windows/`, `bay_of_bengal/results/`
- `arabian_sea/data/windows/`, `arabian_sea/results/`
