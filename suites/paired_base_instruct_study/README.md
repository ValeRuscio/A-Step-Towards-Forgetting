# Paired base/instruct forgetting study

## Models and scope

- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-0.5B-Instruct
- HuggingFaceTB/SmolLM2-360M
- HuggingFaceTB/SmolLM2-360M-Instruct

This is a NEW, separate experiment. It does not alter or resume the reference-frame worker or the completed OLMo/Pythia suite. Qwen2 uses the Qwen2 adapter; SmolLM2 uses the Llama adapter. No Qwen3 classes are required to load these models. Each repository revision is resolved once to an immutable commit and saved in sources/<model>/model_pin.json.

The primary comparison uses IDENTICAL plain-text association prompts for base and instruct checkpoints. No chat template is applied. It asks whether the released checkpoints exhibit different subsequent learning/forgetting dynamics under this fixed protocol. It does not isolate instruction tuning from all other post-training, measure loss of general instruction-following ability, or compare native chat performance.

## Run

1. Upload/extract the entire folder on the GH200, keeping all Python files together. Open Run_paired_models.ipynb in the existing working environment. No dependency upgrade is requested.
2. Set OLMO_SOURCE to the original olmo_association_v1 directory (its config.json supplies the unchanged task and optimizer recipe).
3. Set PREVIOUS to the completed multimodel_paper_v1 result directory, its multimodel_share.zip, or the original frozen_predictors.json file. The runner imports the existing frozen OLMo predictors byte-for-byte, records their checksum and refuses to overwrite them with different fits. It does NOT refit to Qwen/SmolLM results or rerun old OLMo refinements.
4. Run All to create the controls. Click Launch/resume. Stop, Export and New run are buttons and are NOT executed by Run All. Without ipywidgets, the notebook prints manual function calls; no automatic action is taken.
5. Refresh and Show results on demand. Stop is cooperative: wait until alive=False before moving to another GPU study. Launch resumes; New run creates a fresh output without deleting earlier results. Scientific settings changes also use a new directory. Session time/disk budgets may be changed without invalidating completed work.

The notebook checks the known previous reference-frame and multimodel worker directories before starting. Add other active GPU-study output directories to OTHER_GPU_RUNS if needed. This guard does not discover every process on the machine. Do not run these full-gradient workers concurrently on the same GPU.

## Schedule and saved measurements

Defaults: four checkpoints × three seeds × the original B-training duration (128 updates in your source), yielding 1,536 dense update measurements. The original fixed A-training budget is repeated for each checkpoint/seed. The family pair's semantic data and plain-text tokenization hashes must agree. The answer tokens remain the original single-token red/blue protocol; a tokenizer mismatch fails explicitly rather than silently substituting labels.

Before long training, each real checkpoint is audited against its native full-model forward, checked for finite gradients on all unique parameters, and checked for tied-parameter double counting. Bias and normalization parameters are included. Full-parameter FP32 training and eager attention are retained for derivative work. The source Adam coefficients, clipping, learning rate, minibatch schedule and transition policy are reused.

Every natural B update records:
- Full-population A/B task gradients and native history/current Adam channels.
- All four-vector dot products/cosines/norms, globally and per parameter layer.
- Recent vector changes and algebraic dot-product-change decomposition.
- Actual A/B loss change, initial directional prediction and finite remainder.
- Per-entity loss, answer margin and answer-support mass on all original evaluation splits.

These are parameter-gradient measurements, not activation covariance spectra or a new activation-frame intervention. The instruction checkpoints do not come with their historical training optimizer state: measured momentum is created by the subsequent A/B experiment.

PATH_SEEDS IS EXPLICIT: settings['path_seeds'] defaults to [1]. For that seed, each model gets two retrospective validation-selected diagnostic events: largest A_valid increase and nearest-zero A_valid change (not necessarily an ordinary A_test outcome). These receive adaptive finite-update paths and directional HVP/finite-difference audits. Set path_seeds=[1,2,3] before first launch if you want all seeds; this increases cost. Zero path points on unrequested seeds is expected. Completed computations can still have failed numerical audits; inspect path_accounting.json and curvature records.

Session defaults: 11.5 hours, 10-minute reserve, 80% of available session allocated to dense replication, remainder to path diagnostics. When trajectories finish early, paths begin immediately. Remaining work resumes over later sessions; full-queue completion in one night is NOT promised. Three-seed A acquisition can be a substantial part of the cost.

## Comparison and prediction

paired_comparison.json reports A_valid and A_test acquisition, A/B endpoint outcomes, answer-restricted accuracy, large A-loss increases, sign reversals, and matched-seed base/instruct differences when the same number of updates is available. The retention difference is adjusted by subtracting each checkpoint's own A anchor; this still does not make acquisition levels equal.

This is a FIXED-BUDGET acquisition comparison. It is not validation-matched acquisition or tuned learning rates. Report weak acquisition and prompt generalization gaps instead of silently excluding them. A separate validation-selected acquisition-matching study would answer a different question.

The frozen predictors target A_test increases above 0.05 KL. Lead 0 uses current-update gradient information; lead 1 uses the preceding pre-update measurement. Do not conflate those horizons. Raw metrics are per seed; correlated updates are not independent replicates. Existing OLMo selection was exploratory. Save this new protocol before inspecting the new outcomes.

## Files

- paired_suite.py: scheduler, pinned model loading, acquisition, paired report, resume/export.
- multimodel_engine.py: architecture-specific adapters, no unconditional Qwen3 import.
- Run_paired_models.ipynb: quiet button controls and remembered settings/output.
- Other supplied .py files: existing four-vector, numerical path and prediction implementations.
- sources/<name>/adapter_audit.json: native forward/gradient checks.
- REPORT.txt, paired_comparison.json, transfer_metrics.json, prediction_trajectories.png.
- runs/<name>/seedN/measurements.sqlite: complete saved measurements.
- sources/<name>/seedN/anchor.pt: acquired A checkpoint and optimizer for replay. Interrupted A training has one replaceable continuation checkpoint.
- paired_models_share.zip: snapshots of measurements, predictor fits, reports, pins and code. Source model/optimizer checkpoints are excluded from sharing but retained locally.

No raw gradient histories are written. Default output cap is 150 GiB with a 30 GiB free-disk reserve; atomic acquisition checkpoint replacement requires extra room. Cache only the pinned revisions needed. Keep anchors for reproducibility and resume.

## Validation and limitations

Tests use tiny randomly initialized Qwen2 and Llama on CPU, not the full released checkpoints. Forward equivalence, full gradients, exact natural parameter/optimizer preservation, finite paths/HVPs, acquisition resume, the four-case paired pipeline and fixed predictor import are tested. Full-size GH200 duration and real-model outcomes are not claimed. See VALIDATION.txt.

Native chat formatting, activation-frame interventions, new persistent homology, matched-acquisition retuning, and continued repair branches are not part of this runner. Those require separately specified comparisons. This experiment is the focused extension of the completed multimodel four-vector study.
