# Multimodel forgetting: targeted audits and advance-warning tests

This is a self-contained Python + notebook package for the next paper experiments. It **does not require editing the previous notebook** and does not modify the original OLMo results or source checkpoints. It prepares experiments to run on your GH200; it has not already run the full pretrained models.

## Start

1. Extract the ZIP with all its Python files together and open `run_paper_suite.ipynb` in your working GPU environment.
2. Set `OLMO_SOURCE` to the original `olmo_association_v1` source containing `config.json`, `seed*/anchor.pt`, and original forks. Set `FOUR_VECTOR_RESULTS` to your completed four-vector result directory **or its share ZIP**.
3. Run Setup and Launch. Re-run Refresh for progress. Show prints saved reports/plots; Export gives a share ZIP. Stop is guarded by `STOP_NOW=False` so Run All does not immediately stop the worker.
4. Launch again after a pause. Scientific-setting changes produce a new directory; runtime budgets can change without restarting the science. The notebook remembers the last directory across kernel restarts.

Use the existing working CUDA PyTorch installation. The adapters require Transformers >=4.51,<5 for Qwen3, and were tested with 4.57.6 / PyTorch 2.8.0. Do not blindly replace your CUDA build. If upgrading Transformers is necessary, the original OLMo checkpoint-reconstruction hash checks must still pass before cached path positions can be reused.

## The three parts

### 1. Resolve, or explicitly retain, the difficult OLMo audits

Refinement targets seed1/update021 and seed2/update011. A separate SQLite copy retains their saved positions. The point cap rises from 129 to **1,025 per task/path**, and maximum subdivision depth from 10 to 16. Local loss closure and convergence of history, current, rounding, magnitude, orientation, and joint integrals are checked separately.

Full-population slope audits use widths 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625. Acceptance requires two adjacent widths to agree with autodiff and each other. At each event, curvature is checked at the **original audited site** and a site selected from the refined path's largest absolute directional-derivative secant. Full-population Hessian contractions and their parameter-layer contributions are retained. Shrinking-width curvature attempts are saved individually and can resume.

**A larger budget does not guarantee resolution.** Remaining failures are reported as unresolved. No tolerances are silently relaxed, no failed case is dropped, and no precision change is silently mixed into the original FP32 experiment. These are retrospective numerical checks, not new independent observations. `finished` means the configured attempts ended; `resolved` means the checks passed. The original single-width curvature rows are excluded from the refinement copy's `curvature` table to prevent presenting old checks as new; original files stay untouched.

### 2. Test whether pre-update changes give advance warning

Before new-model data are collected, the suite freezes a protocol and predictors fitted **only to the original OLMo trajectories**:

- Primary outcome: the next targeted update increases full-population A-test mean KL by **more than 0.05**. This is an operational large one-step increase, not automatically a persistent catastrophic forgetting event.
- **Lead 1 (primary):** features from pre-update measurement t−1 predict the loss increase at update t. This supplies a full intervening update of lead time.
- **Lead 0 (secondary):** measurements from just before update t predict that update. The current training minibatch gradient is already available; this is a current-gradient-informed diagnostic, not an earlier warning.
- Features use only pre-update losses, norms, cosines, channel projections, recent vector-change decompositions, and coarse parameter-layer thirds. Recent loss-change controls are computed from consecutive *pre-update* losses, so they were already observed when the feature vector was measured.
- No actual-displacement remainder, post-update loss, within-update path feature, curvature audit, or future update feature enters the predictor.
- Baselines: loss/time/recent-loss history; that baseline plus local A projection; static vector geometry; static geometry plus vector changes. Fixed ridge logistic regression C=0.1, training-fold-only standardization, no hyperparameter search. A small NumPy damped-Newton implementation is included; scikit-learn is not required.
- Original OLMo evaluation leaves one **entire seed** out. These tests remain exploratory because the OLMo results informed the project and protocol.
- Fits on all original OLMo seeds are frozen and transferred to Pythia and Qwen without retraining, threshold tuning, or standardizing on their data. Those evaluations are prospective only if the protocol is frozen before inspecting new-model outcomes.
- AUROC, average precision, Brier score, event counts and prevalence are reported separately by model, seed, predictor and lead. A single-class evaluation yields undefined discrimination metrics, not invented zeros. No independent-update p-values or inflated confidence intervals are reported.

These predictors require labeled prior-task examples to compute A loss and gradients. They are research diagnostics with prior-task access, not data-free or inexpensive deployment monitors. The diagnostic gradients do not enter the training optimizer.

The positive hypothesis is **incremental predictive value of changes beyond static geometry and recent-loss baselines**. If it fails or varies by architecture, that is retained. Raw dimensions and calibration can differ across models; transfer failure is not by itself proof that no common mechanism exists. Three seed trajectories are three downstream task/data realizations, not hundreds of independent replications or independently pretrained models.

### 3. Replicate on additional pretrained models

Defaults:

- `EleutherAI/pythia-70m`
- `EleutherAI/pythia-410m`
- `Qwen/Qwen3-0.6B-Base`

Qwen3's small model is **0.6B**, not 0.5B. The **base** model avoids adding chat/post-training differences to this comparison. Official model pages: https://huggingface.co/EleutherAI/pythia-70m , https://huggingface.co/EleutherAI/pythia-410m , https://huggingface.co/Qwen/Qwen3-0.6B-Base . A requested Hub revision is resolved to an immutable commit once and saved before downloading weights. Only that revision is loaded on resume; no pretraining-checkpoint sweep is downloaded.

For each enabled model and seeds 1,2,3, the original A-training budget, B-training length, optimizer, learning rate, clipping, minibatch schedule, target probabilities and prompt texts come from the original OLMo source config. Usually this is 600 A updates followed by 128 B updates. Red/blue answer semantics are fixed and single-token support is checked per tokenizer. Bare association prompts are used, with no chat template, generation or thinking mode.

All models are evaluated in FP32 with deterministic settings and explicit eager attention for new-model runs, allowing second derivatives. Pythia uses a GPT-NeoX adapter including fused attention matrices and biases. Qwen includes its normalization and tied embedding/readout parameters; shared parameters are counted once. Full-vocabulary target KL and its unclipped full-parameter task gradients are used. The current Adam channel retains the original training-gradient clipping. No OLMo-only Q/K/V layout is assumed.

**Acquisition is a prerequisite to interpreting forgetting.** The suite records all four populations before and after A training and marks whether A-valid KL is <=0.25, fixed in advance. Runs failing this criterion are retained and flagged in transfer tables, not discarded or described as successful retention. The budget is not tuned using B or test performance. The shared learning rate is a controlled protocol, not a claim that it is optimal or equally stable for each architecture. Record the full A/B trajectories when making retention claims.

Every B update gets the original four-vector measurements, recent vector changes, layer-specific projections, and actual before/after losses. Thus, with the original 128-step source, the three new models generate **1,152 dense update records**.

To bound the deeper work, **seed 1 of each architecture** also gets two finite paths: its largest A-valid increase and its nearest-zero A-valid change, selected only after that full trajectory exists. These are retrospective diagnostics, not predictor-training data. Selection by validation prompt form does not make the resulting test measurements independent samples. The default adds **six finite paths**, each with A/B population accounting and one selected curvature position. Its 129-point-per-task cap and single-width curvature checks match the earlier baseline experiment; flags are retained. Further difficult cross-model cases can be targeted later rather than automatically expanding every job.

## Time, memory, and storage

The default session is 11.5 hours with a 10-minute reserve. Session allocations are 25% original-path audits, 60% new-model acquisition/trajectories, and the remaining time for cross-model path diagnostics. Unused earlier-stage time is available before later absolute deadlines. If the audit stage uses its allocation, replication still gets a chance that night. Work resumes by completed unit.

New-model trajectories are scheduled round-robin by model and seed, 16 measured updates per visit. Model loading and A acquisition can still be substantial. **This entire expanded queue is not promised to finish overnight**; expect to use repeated sessions and inspect measured throughput. If time is short, disable Qwen before launching or allocate more time to replication. Changing which models are enabled creates a new output because it changes scientific settings.

The suite needs substantial GPU/host RAM for full gradients, FP32 weights, optimizer states and Hessian-vector products. The GH200 is the intended target; only tiny CPU architecture tests have run locally. A training progress checkpoint is replaced every 50 steps and at cooperative interruption. Each completed A anchor keeps weights and optimizer state needed for exact B replay. For the three models × three seeds, expect **tens of GiB of checkpoint storage** plus temporary atomic-save headroom. No raw activation or full-gradient array dumps are written, and only one cached pretrained revision per model is requested. Default suite storage guard: 200 GiB; free-space reserve: 30 GiB. Nothing from previous research is deleted.

## Outputs

- `REPORT.txt`, `transfer_metrics.json`, `prediction_trajectories.png`.
- `prediction_protocol/frozen_predictors.json`: protocol, fixed fits, original-data hash and exploratory held-out-seed scores. Do not refit after looking at transfer outcomes and call the results prospective.
- `refinement/REPORT.txt`, path accounting/audits and `measurements.sqlite` with `curvature_refined` rows and every shrinking-width attempt.
- `sources/<model>/model_pin.json`, `config.json`, `seed*/acquisition.json`, `data.json`, `anchor.pt` and temporary `A_progress.pt`.
- `runs/<model>/seed*/measurements.sqlite`: full saved trajectory, path and curvature records. Same four-vector JSON schema as before. Layer Hessian contributions include cross-layer couplings.
- `predictions/<model>/predictions.json` and `prediction_metrics.json`: scores, outcomes and feature/target timestamps for auditing lead time.
- `multimodel_share.zip`: all saved measurements, code, protocols, reports and pins. **Checkpoints are excluded from the share ZIP**, but remain in sources. Full raw vectors are not stored; reproducing them requires those checkpoints.

No new PH/RMT campaign is included: this package focuses on the three requested gaps so it can support the paper deadline. It does not infer phase transitions, topology changes, reference-frame causality, or a universal forgetting mechanism from prediction accuracy.

## Validation

`test_multimodel.py` checks tiny OLMo/GPT-NeoX/Qwen3 full-forward agreement, all trainable gradients, unchanged natural weights/Adam state, finite paths/HVP audits, acquisition pause/resume, predictor timing, and numerical classifier metrics. `test_suite_pipeline.py` exercises a complete tiny pipeline through import, refinement, new-model training, prediction and export. The included original tests remain available. See `VALIDATION.txt` for checks actually run before packaging.

Dependency repair: Qwen3 classes are imported only for Qwen3. If unavailable, Pythia work continues and Qwen is listed in deferred_models.json; final status remains dependency_blocked until Qwen completes. Existing environments are never upgraded automatically.
