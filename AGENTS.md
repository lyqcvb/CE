# Repository Guidelines

## Project Structure & Module Organization

This repository implements an EEG brain-network analysis pipeline for HC, MCI, and AD groups. Source code lives in `src/`. The pipeline is ordered as `step00_bandpass_filter.py` through `step12_fc_heatmaps.py`; `step03_connection_strength.py` computes `MeanConn_*` between FC construction and graph metrics. `brain_metrics.py` is a compatibility facade for `import brain_metrics as bm`.

Input EEG data is expected under `data/raw/`. Generated intermediate files go to `output/`, and final summary tables/figures go to `results/`. Method notes and formulas are documented in `docs/algorithm_summary.md`. There is currently no dedicated `tests/` directory.

## Build, Test, and Development Commands

Install dependencies with:

```powershell
uv sync
```

Run the main pipeline:

```powershell
uv run python src/step09_batch_pipeline.py
uv run python src/step09_batch_pipeline.py --group HC --band "1-4(Delta)"
uv run python src/step09_batch_pipeline.py --fc-only
```

Generate filtered band data and downstream outputs:

```powershell
uv run python src/step00_bandpass_filter.py --skip-existing
uv run python src/step10_results_postprocess.py
uv run python src/step11_statistical_analysis.py
uv run python src/step12_fc_heatmaps.py
```

Use `uv run python -m compileall -q src` as a quick syntax/import smoke check.

## Coding Style & Naming Conventions

Use Python 3.12, 4-space indentation, type hints where practical, and NumPy-style array naming. Keep process modules named by pipeline order (`stepNN_description.py`) so execution flow stays obvious. Preserve existing public function names exported by `brain_metrics.py`; other scripts depend on them.

## Testing Guidelines

No formal test framework is configured yet. For changes to metrics, add small deterministic smoke tests or scripts that exercise LOFC, HOFC, aHOFC, scalar metrics, rich-club, and target attack on synthetic matrices. Prefer fixed RNG seeds and compare shapes, finite values, and known edge cases.

## Commit & Pull Request Guidelines

The current Git history uses short messages such as `restrcture` and `1.0`; improve on this with concise imperative messages, for example `Refactor metric modules by pipeline step`. Pull requests should state the affected pipeline stage, commands run, and whether outputs in `output/` or `results/` were regenerated. Include screenshots only for visualization changes.

## Agent-Specific Instructions

Do not commit generated data unless explicitly requested. Avoid rewriting unrelated files in a dirty working tree. When changing formulas, update `docs/algorithm_summary.md` in the same change.
When a code change requires rerunning experiments, update the relevant `docs/` documents in the same change, including the algorithm document and the matching results writeup(s) such as `docs/结果文档-512.md`. If input/output contracts change, also update `docs/script_io_formats.md`.
