# Reproducing a run

## Execution environments

Each suite is independent. Keep its files together and execute from its directory. Do not flatten these folders: identically named modules such as `study.py`, `tasks.py` and `multimodel_engine.py` differ between studies.

For reproducing published numbers, prefer the `recorded_runs/<run>/code` version and recorded settings/environment, rather than assuming the current distributed default is identical. Notebook wrappers and READMEs are in the corresponding distributed suite. Review the imported module/API before substituting recorded source. An exact historical environment may differ from the newer Qwen environment.

Requirements with ranges describe compatible dependencies, not an exact environment lock. Available `environment.json` records are preserved for the original versions. Do not upgrade a running experiment's environment. For new Qwen3 GH200 runs, `suites/qwen3_history_replication/Run_Qwen3_History.ipynb` includes isolated setup, CUDA verification and resume controls. Its setup targets Linux GH200/ARM64; this is not a universal CUDA installer.

The narrowly scoped cuSPARSELt ARM64 wheel-tag repair is included with that suite. It checks the exact known discrepancy, library architecture/loading and GPU differentiation; it is not a general instruction to ignore dependency errors.

## Models, data, and input paths

Model/dataset downloads need network access on first use. Preserve pinned revisions, sampled example identities and tokenization. Settings files may contain original `/home/ubuntu/...` paths. Change only local input/output/interpreter locations when relocating, and retain a record of the relocation. Never reinterpret a missing source path as an empty dataset or a new training run.

Some early suites require prior original experiment directories. Their result-share ZIPs generally omit checkpoints and cannot necessarily substitute for those directories. History replay specifically needs saved source settings, data, environment and results.sqlite. The complete original remote run is not bundled here. When supplying supplementary data, retain these input records separately with hashes.

## Launch, pause and resume

Use the selected notebook's Setup/Launch-resume/Status/Stop/Results/Export cells; module APIs differ across suites. Status refresh does not start a job. Cooperatively stop and wait until the worker is inactive before modifying operational settings. Resume the same run; restart creates a fresh run. Session time limits are not total runtime estimates. Hessian/path studies may need several sessions.

Do not launch all suites at once. They can compete for GPU memory and their source locks enforce only their own pipeline relationships. Storage reserves and output caps exclude model download caches in several suites. Do not delete source data or optimizer checkpoints needed by an active replay.

## Verification levels

1. `python verify_archive.py`: file integrity, Python parsing, notebook structure; no GPU experiment.
2. Suite-local tests: consult local README/VALIDATION. Many use tiny CPU models and need the suite's ML dependencies. Run in separate processes from each suite directory; do not import all suites into one interpreter.
3. Recorded-run audits: exact endpoint hashes, channel/age algebra, finite differences, quadrature, curvature and acquisition gates. Passing an algebra check does not imply derivative validity. A capped or unresolved path is retained as unresolved.
4. Scientific reproduction: compare full saved summaries on the same populations and selection rules, including failed gates and incomplete cases, using the recorded configuration.

## Interpretation conventions that must remain fixed

- Native history/current channels share the updated Adam denominator and retain their actual magnitudes.
- History ages are contributions under the current preconditioner, not causal deletion experiments.
- Confusion plus leakage equals total loss for the declared answer set.
- Replay's 32-example A-test derivatives/outcomes differ from 128-example source endpoint means.
- Uniformly sampled and validation-enriched events have different uses; do not pool them as a prevalence estimate.
- Current-update diagnostics differ from previous-update forecasts. Use the timeline specific to each suite.
- Gradients, prompts, updates and paired answer conditions are not independent training seeds.
- The cumulative remainder includes numerical/displacement residuals; it is not automatically mixed curvature.
- Qwen text natural trajectories, history replay and paths can have different completion statuses.
