# Suggested Git workflow

1. Pull the newest team repository.
2. Copy this folder into the repo as `03_analysis/`.
3. Keep Sun's full handoff outside normal Git.
4. Run the analysis and inspect every generated figure/table.
5. Commit `03_analysis/run_analysis.py`, `requirements.txt`, `README.md`, and the small `analysis_outputs/` tables/figures you actually use.
6. Do not commit model weights, caches, checkpoints, or the ~500 MB prediction folder unless the group deliberately uses Git LFS.

Suggested commit message:

```text
Add Part 03 robustness analysis and report figures
```
