# OLMo overnight: routing, content, geometry and forgetting

Complete Python code and a quiet notebook for the original OLMo association campaign. Default session: **11.5 hours on one GPU**, with a six-minute reserve for reporting and export. Designed to use the existing GH200 FP32 environment without changing the original training semantics.

## Start tonight

1. Extract this package into a new folder on the GPU machine. Keep both `.py` files beside `Run_structure_study.ipynb`.
2. Open that notebook using the environment that ran the previous OLMo study.
3. Run **Setup**, **Configure**, and **Run / resume**. The source is your ORIGINAL association campaign, usually `/home/ubuntu/1/runs/olmo_association_v1`, containing `config.json`, per-seed `data.json`, `anchor.pt`, and `forkNNN/fork.pt`. A previous results ZIP is not the source.
4. Leave it running. Rerun **Refresh status** or **Show results** manually whenever you want an update. There is no continuously printing notebook cell.
5. At session end, **Find / regenerate the results ZIP** prints the archive path and supplies a notebook download link where Jupyter permits it.

All complete source seeds are discovered automatically. The default event order is 21, 20, 47, 46. A preceding fork must exist for each requested event. The source checkpoints are read only. New continuation experiments live in the separate output directory.

## Stop, resume, restart

- **Stop:** set `STOP=True` in the labelled cell and execute it. It terminates the tracked launcher and worker process group.
- **Resume:** execute Run / resume again. An already-active run is left running. A stopped/budget-paused compatible run gets a fresh session budget and continues unfinished units.
- **Restart:** set `RESTART=True` in its labelled cell. The tracked run is stopped and a new output folder is created, preserving previous results.
- `overnight_session.json` beside the notebook remembers the active settings/output location across kernel restarts. Run Setup and Configure again to recover that location.

Changes to settings, code or dependency versions select a fresh output folder when you launch an inactive run. Changes to fingerprinted source inputs are refused within an existing design. No manual lock-file deletion or patch cells are needed.

The time budget covers a session, not the entire planned campaign. `budget_paused` is a valid outcome with useful saved results. The queue deliberately contains more work than one night may accommodate. Job counts and output presence distinguish completed, pending and failed work. A missing result is not a null finding.

## What is collected

### 1. Dense actual-update paths and exact slopes

For every measured event, the code replays the original saved optimizer state and original scheduled B minibatches. It measures the actual parameter displacement, native Adam history/current numerator channels, and full A-test gradient contribution.

It then probes fractions of that fixed displacement. The default initial grid has 65 fractions: 0.025 spacing across the update plus 0.005 spacing over 0.50–0.65. Up to 12 additional points target finite-interval nonlinearity and slope reversals. This fixes the earlier sign-change-only refinement's blind spot around the steep loss rise.

Every fraction records:

- full A_valid/A_test/B_valid/B_test losses for every source entity;
- exact decomposition into relative-answer KL and probability outside the two-answer support;
- full A-test AND B-test directional slopes;
- slope contributions and gradient/update norms for every named parameter, annotated by layer and functional channel;
- two finite-difference checks and individual entity finite-difference slopes.

The parameter channels are QK, OV, MLP gate, MLP content, normalization, and embeddings/readout/other. These are positions in the computation graph, not independent computational systems. Path fractions are virtual measurements of one actual update, not additional training steps. Diagnostic probes restore the actual endpoint and do not advance optimizer moments.

### 2. All-layer hidden geometry and persistent homology

Nine fractions receive deeper measurements: 0, 0.234375, 0.50, 0.55, 0.575, 0.60, 0.625, 0.71875, 1. All decoder-block outputs and final hidden state are measured, using all entities in both tasks and both validation/test prompt forms.

Saved data include final-position hidden vectors, answer/support Jacobians, hidden and decoder norms, readout-relative angles, exact final-margin decomposition, and task-sensitive displacement. Ordinary Euclidean and frozen A-anchor answer/support Fisher distances retain fixed scales across the path.

The PH computation includes all-entity H0/H1 diagrams, paired bottleneck distances, target-labelled component mixing, nearest-same/opposite target gaps, validation/test same-entity distances, and 32 label-permutation controls per cloud. Labels annotate topology; they do not change unlabelled persistence.

The frozen metric is calibrated with A_valid at the original A anchor, once per seed. It coarsens the output to answer0/answer1/other and can be degenerate. It is not full-vocabulary Fisher, not a loss Hessian, and not a geodesic through a changing metric. At intermediate blocks, Jacobians hold the other token positions fixed. Entities are shared across prompt forms, so held-out prompts are not unseen-entity generalization.

All-layer attention geometry at these fractions includes audited reconstructed distributions, entropy, first-key Q/K alignment and relative spatial phase. Phase uses a fixed first-key reference, not a temporal optimizer phase estimate.

### 3. Parameter-channel factorials and every-layer restoration

At fractions 0.50, 0.575, 0.625 and 1, the code crosses earlier/later parameter groups in three two-by-two designs:

- QK versus OV;
- MLP gate versus MLP content;
- selection (QK + gate) versus content (OV + MLP content + embedding/readout/other).

All parameters outside the crossed groups remain at the same fraction's background; normalization remains separate. Each design records the four task losses and the finite interaction `L11 - L10 - L01 + L00`. This measures non-additivity, not a unique mechanism.

Every layer also receives restorations of QK, OV, gate, MLP content, and its entire block, with both the 0.625 background and full endpoint. The corresponding source pre-update weights are restored in each intervention; all other weights retain the common background. These controlled counterfactuals test parameter dependence without pretending that frozen QK implies fixed attention.

### 4. Audited routing/value activation interventions

The model captures attention operands before the update and at fraction 0.625, for all tokens, all layers and all four task/prompt splits. Raw arrays retain attention probabilities, values, actual pre-output-projection activations, normalized attention inputs, and projected Q/K vectors.

At every layer it evaluates:

- earlier routing with later values;
- later routing with earlier values;
- earlier routing and earlier values;
- reconstructed later routing/values (sham);
- two random activation perturbations matched to the earlier-both displacement norm.

The patch replaces all query positions BEFORE the current attention output projection. O and the downstream network remain fixed. Thus this is a routing/V-content experiment, not a claim to isolate the entire OV parameter channel.

The exact identity `delta(PV) = deltaP V0 + P0 deltaV + deltaP deltaV` is recorded with reconstruction errors. Reconstructed `P@V` must agree with the native output, and the sham must reproduce unpatched loss within 1e-4, before a comparison is marked audited. An output audit does not uniquely verify every probability.

The more expensive individual-head stage is queued separately, so it cannot block basic coverage. It tests the four heads with largest A_valid attention changes per layer. Random head controls perturb the same head support and match activation norm. Head selection uses validation prompts; all test entities remain in the evaluation. Patches can create internally inconsistent states, so sham and matched controls are essential. Causal effects of patches still require replication and interpretation.

### 5. New controlled training continuations

From the same original pre-update state, seven branches run up to 16 actual B updates using identical scheduled minibatches:

1. Original Adam continuation.
2. Reset first moment only.
3. Reset second moment only.
4. Reset both moments while retaining the original Adam timestep.
5. Fresh optimizer, including a fresh timestep.
6. Apply 0.375 of the first parameter displacement, retaining its original moment update; subsequent steps use original Adam settings.
7. Use 0.25 of the original learning rate throughout the branch.

Every step evaluates all four splits and retains per-entity loss components and clipping statistics. These are new model experiments. The fractional branch is an exploratory fixed choice motivated by the previously inspected event, not an independently selected optimal learning rate.

A numerical failure in a branch is recorded and the other branches continue. A branch interrupted by the time budget is replayed from its source state on resume; partial trajectory measurements remain available. Sixteen steps are short continuations, not proof of long-term retention.

### 6. Gradient/momentum geometry

At fractions 0, 0.575, 0.625 and 1, all parameters receive full A-test/B-test gradient norms, cosine with the reconstructed native pre-update first moment, actual-update projections, and history projections under the fixed updated Adam denominator.

These distinguish parameter channels and layers. They measure alignment with optimizer memory; they do not establish a temporal oscillation or frequency-domain phase. The pre-update moment reconstruction inherits FP32 rounding. Training-minibatch clipping is recorded separately from the unclipped evaluation gradients.

### 7. Every-matrix spectrum, RMT and subspace motion

At fractions 0 and 1, every unique two-dimensional weight matrix is measured, with tied aliases recorded. The outputs include full Frobenius norms, randomized leading SVD with residual audits, fixed-subsample covariance spectra, Gaussian/shuffled RMT controls, sampled-row PH, leading subspace angles, and leading radial coefficients of the actual displacement.

The leading radial certificate is truncated and first-order; it does not establish a finite isospectral update. Singular-vector interpretations are sensitive to degeneracy. The Marchenko–Pastur edge is a descriptive asymptotic iid null for a normalized subsample, not a significance test or full-matrix spectral law.

### 8. All-entity, channel/layer-resolved task-coordinate NTK

At fractions 0, 0.575, 0.625 and 1, the complete trainable-parameter Jacobian of answer/support coordinates is measured for every A/B test entity. Two independent CountSketch projections per parameter channel/layer produce approximate kernel blocks. Exact diagonals audit the sketches; disagreement is saved.

The data include A/B gradient alignment, infinitesimal SGD projections, kernel rank summaries, channel/layer kernels, and sketched per-entity loss-gradient coherence. This coherence is not winner-score sign coherence and not minibatch or temporal coherence. The NTK concerns two task coordinates, not every vocabulary logit, and does not predict a historical Adam update directly. Each coordinate is checkpointed to disk so this expensive stage can resume.

## Priority and resource policy

The initial priority is dense paths for seed 1 updates 21/20, then update 21 for other available seeds. All-layer geometry and factorials for seed 1 follow. Continuations, gradient/momentum diagnostics, matrices, whole-layer activation patches and NTK for its primary event follow before the extended replication/head queue.

Additional seeds, updates 47/46 and remaining stages stay queued. Nothing is silently labelled completed when time runs out. Default work stops six minutes before the 11.5-hour session limit; an unresponsive worker is terminated and completed units remain intact. Reports and the share ZIP are then produced. This is a practical wall-clock budget, not a guarantee against operating-system/driver stalls or unusually slow export.

GH200 hardware and actual peak GPU allocations/job durations are recorded. One worker uses one GPU at a time. Original FP32, clipping, minibatches and attention backend are retained; no precision downgrade or optimizer substitution is performed silently. Host memory can hold several parameter-sized tensors during gradient/momentum analysis; use the existing GH200 host environment. Real GH200 throughput and full campaign completion within one session have not been measured locally.

Raw arrays may occupy many GB across events and seeds. The worker pauses if free disk falls below 8 GB. The automatic share ZIP includes JSON, CSV, text, logs and figures, while raw NPZ arrays remain in the results directory. `export_arrays=True` in settings, or `INCLUDE_RAW_ARRAYS=True` in the export cell, includes them in a larger archive. Check `EXPORT_SCOPE.json` inside any archive to see what it contains.

## Output map

The notebook always prints the current output directory. Inside it:

- `status.json`, `jobs.json`, `session_budget.json`: actual status, queue and session budget;
- `hardware.json`, `design.json`, `settings.json`, `source_config.json`, `model_config.json`, `source_data_seedN.json`: hardware, code/input fingerprints and the full source/model/task configuration;
- `overnight_report.txt`, `overnight_summary.json`, `overnight_paths.png`: compact readouts;
- `intervention_summary.json/.csv`, `patches_*_effects.png`: audited activation-patch effects;
- `results_share.zip`: automatically generated archive to download and share;
- `calibration_seedN/`: frozen all-layer A-anchor calibration;
- `path_seedN_stepNNN/`: dense losses, per-parameter slopes, finite differences;
- `geometry_seedN_stepNNN/pointNNN/`: hidden arrays, Jacobians, labelled topology, attention and path features;
- `factorial_seedN_stepNNN/`: parameter counterfactuals, interaction values and layer restorations;
- `patches_seedN_stepNNN/`: input-matched raw captures, whole-layer and head patches, audits;
- `branches_seedN_stepNNN/`: seven new continuation curves and plots;
- `gradients_seedN_stepNNN/`: gradient/memory comparisons;
- `matrices_seedN_stepNNN/`: matrix spectra, RMT, row topology and subspace comparisons;
- `ntk_seedN_stepNNN/`: resumable coordinate shards and assembled kernels;
- `heads_seedN_stepNNN/`: scheduling/completion records; head results themselves are in the matching patches folder.

Each job has its own progress, log and completion marker. Worker failures have a traceback file. Reports select the largest completed whole-layer A_valid rescue only as a descriptive summary; this is not multiplicity-corrected significance or independent validation.

## Local validation

See `VALIDATION.md`. The complete campaign was tested on a tiny original-OLMo CPU model, including every stage, independent replay, interventions, NTK checks, automatic export, time-budget pause, resume, restart and stop. The actual GH200/OLMo-1B campaign has not been run here.

This package collects a broad, reproducible measurement set. It does not compute an exhaustive full Hessian, full-vocabulary NTK or every possible intervention; it does not infer causality from PH/RMT alone. The original checkpoint/trajectory functions remain in the same complete module for compatibility, but the default overnight mode exclusively executes the documented overnight queue.
