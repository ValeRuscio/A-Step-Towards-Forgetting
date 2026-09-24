# A Step Towards Forgetting — reproducibility source collection

This archive collects the available experiment implementations and notebook wrappers used during the project. It separates runnable distributed packages from code snapshots exported with actual results. It is a source release, not a claim that every planned experiment finished or that every result is reproduced by this archive alone.

## Start here

1. Read `PAPER_TO_CODE.md` to select the experiment relevant to a claim.
2. Read `REPRODUCING.md` for environment, saved-input and replay requirements.
3. Read `EXPERIMENTS.md`, then the selected suite's own README and notebook.
4. Run `python verify_archive.py` to verify integrity, Python syntax and notebook structure without installing ML libraries or launching a GPU experiment.

## Contents

- `suites/`: 25 independently packaged experiments, with Python implementations, quiet notebook wrappers, available tests, requirements, calibration inputs and frozen predictors. Early exploratory suites are retained for completeness; they are not all main-paper evidence.
- `recorded_runs/`: Python code and available configurations/environment metadata extracted unchanged from nine user-supplied result archives. These are the strongest available provenance for those specific results. They deliberately remain separate from distributed packages.
- `historical_repairs/`: separately delivered compatibility/reader repairs, not silently applied to recorded code.
- `PROVENANCE.json`: original archive names, SHA-256 hashes, original members and copied-file hashes.
- `SUITES.json`: machine-readable suite catalogue.
- `SHA256SUMS`: release file hashes.
- `VALIDATION.json`: checks actually performed when assembling this archive.

No model checkpoints, credentials, complete run databases, or full downloaded datasets are included. Necessary compact frozen calibration inputs already shipped inside individual suites are preserved. Paths in historical settings are original machine paths; configure local input/output locations as explained in the notebook. Never rerun into an existing published result directory with changed scientific settings.

## Scientific scope

Distinguish original registry studies, later natural-text studies, matched-history interventions, observational history replay and the ongoing Qwen text replication. They use different seeds, populations and selection rules. The Qwen3 registry results and Qwen3 text/history replication are different experiments.

Exact native replay uses the recorded model revision, data order, optimizer computation and environment, and checks endpoint hashes. Cross-hardware numerical equivalence is not promised. Do not bypass failed replay or derivative checks.

This release was assembled while later studies were still running. See `RELEASE_LIMITATIONS.md`. In particular, included OPEN-1B, Shampoo or Muon pilot code is not evidence that those experiments completed or support manuscript claims.

No new software license is assigned in this collection. Before public redistribution, the authors should select a license for their original code and retain applicable third-party licenses and dataset/model terms. This package does not redistribute pretrained weights or dataset corpora.
