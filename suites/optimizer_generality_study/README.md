# Optimizer generality, timing and answer-token controls

Extract the complete ZIP into a NEW folder. Open **Run_Optimizer_Generality.ipynb**.
The notebook selects your existing compatible GPU Python environment, launches a
background worker quietly, and displays progress only when you rerun the status
cell. Keep all Python files together. No prior research directory is modified.

## A finding already available — read before writing the paper

The package includes a compact, immutable extract of the four existing natural
trajectories (Pythia-410M registry seed11; text seeds11/12/13), with source hashes.
The timing analysis has already been run; see `existing_timing_results/REPORT.txt`.
At the decision immediately before update d, the latest completed A-validation
loss increase from update d-1 is available. Its raw AUROC for an A-test increase
>.05 nats is **.850907/.852176/.940476**, versus previous-update projection
**.818311/.846523/.927154**, for text seeds11/12/13. The earlier fitted loss-history
baseline understated this comparison. No score is refitted or sign-flipped here.
This narrows the advance-warning claim. It does not invalidate the separately
controlled, norm-matched history effects. The present analysis is retrospective.

## Queue, default settings

1. Existing-record timing/persistence analysis, CPU only, no model download.
2. SmolLM2-135M **base**, a small non-Pythia Llama-family transformer: three LR
   candidates for each of SGD, momentum SGD, Adam beta1=0, and ordinary Adam.
   Then three fresh confirmation seeds, 21/22/23, for each selected optimizer.
3. Pythia-410M: its OWN identical-budget LR calibration and the same three fresh
   confirmation seeds. SmolLM learning rates are not blindly transferred.
4. SmolLM2 answer-token control: Adam, three seeds, paired counterbalanced shared
   versus disjoint answer codebooks. Optional; enabled by default. To include
   Pythia as well, add `pythia410m` to `label_models` before the first launch.
5. SmolLM2 appendix pilots: three LR candidates each for blocked Shampoo and FP32
   Muon hybrids, then one preliminary confirmation seed31 per optimizer.

Default: **62 trajectory jobs**, including calibration, main comparisons, labels
and appendix. Primary main runs: 2 models x 4 optimizers x 3 seeds =24. This is a
MULTI-SESSION queue, not a claim that all62 jobs fit one night. Each launch has an
11.5-hour cooperative budget. Run Launch/resume again to continue. Shader/matrix
operations cannot be interrupted until they return. Changing hours or storage
budgets is allowed on resume; scientific settings require a fresh output folder.

Calibration: A300, B64 updates. Main: A600, B128 updates. Appendix confirmation:
A600, B64. Microbatch2, accumulation2, clip1, FP32 weights/optimizer, FP64 loss and
scalar reductions, no autocast, zero weight decay for ALL methods. Dropout is
inactive to make local derivatives/replay deterministic. Same deterministic
minibatch order within seed across optimizers. Primary momentum=.9, beta2=.99,
epsilon1e-8. All parameters are trained. Model revisions resolve once and remain
pinned in `resolved_assets.json`; datasets use the previous fixed revisions.

Fresh seeds also change dataset sampling and minibatch order. They are not merely
weight-initialization seeds: pretrained weights are identical within a model.
This compares two transformer families and sizes, not isolated architecture or
scale effects, and not recurrent/state-space architectures.

## Calibration protocol — no held-out test selection

Raw text hashes assign 20% of each corpus to the development pool and80% to the
confirmation pool. Text-identical records cannot cross the pools. Each seed draws
train/validation/test records from its assigned pool; source train/test splits
remain separate. Calibration executes only train and validation evaluations;
test rows are never read by its objective. Within-pool seeds may share examples,
so do not treat their datasets as mutually disjoint. This does not establish that
examples were absent from original model pretraining.

Per task:1024 train /32 validation /128 test examples. SST-2 sentiment -> AG News
topic classification. Maximum prefix256 tokens. Answer labels must be distinct
single tokens; incompatible tokenizers are refused rather than silently relabelled.

Each model/optimizer has three candidates:
- SGD: .001, .01, .1
- Momentum SGD: .0001, .001, .01
- Adam beta1=0 and Adam: 5e-6, 2e-5, 8e-5
- Appendix blocked Shampoo: .0001, .001, .01
- Appendix FP32 Muon: .005, .02, .08

A candidate must improve A-validation loss by at least .1 nat and reach at least
.60 restricted-answer accuracy after A training. Among eligible candidates,
minimize the mean over saved B-phase evaluation checkpoints of:

    B_valid NLL + max(A_valid NLL - A_anchor_valid NLL, 0)

Both terms have fixed coefficient1. Tie -> smaller LR. This is a predeclared
retention/acquisition objective, not a universal optimizer ranking or exhaustive
hyperparameter search. If no candidate qualifies, confirmation for that optimizer
is explicitly skipped; the code never calls failure to learn "good retention".
Grid failures and skipped cases are saved. Do not retune on confirmation test data.
All selected learning rates and all candidate outcomes are retained.

Every main optimizer trains BOTH A and B using its own state. A history is kept
at the boundary. There is no Adam-to-SGD state conversion. Different optimizers
therefore produce different A anchors. For history interventions, comparisons
instead start at the SAME weights and native optimizer state within a trajectory.

## Measurements

Every main B update records:
- Exact full32-prompt A/B-validation gradients; actual assigned parameter update.
- Task-relative projection/cosine, update and A-gradient norms; conventional
  population-gradient and training-minibatch interference.
- Layerwise projections/norms, global clipping factor, full A/B validation/test
  losses and per-example loss, confusion, leakage, restricted/full accuracy.
- Initial A-valid linear prediction and actual finite-step remainder, on the SAME
  population. Do not compare A-valid projections to A-test loss changes as though
  they used the same gradient.
- SGD, momentum SGD, and Adam native history/current channels and sum-roundoff
  residual. No fictitious additive split is supplied for Muon or the Shampoo hybrid.

At predeclared B updates8 and32, all main seeds with history get immediate
preserve-vs-reset probes: reset vector memory, reset with global first-step norm
matched, reset with transformer-block norms matched. The embedding/readout/other
parameters form one additional block. Second moments, Shampoo accumulators and
optimizer clocks are unchanged. Actual assigned norms are audited at rtol1e-3,
atol1e-10. Nothing is trained further along these probe branches. They establish
immediate consequences, not long-term prevention. Natural training resumes from
its original post-update model AND optimizer state.

At these same steps for seed21, and appendix seed31, saved A-validation paths use
9 points, nested Simpson closure/error estimate, and an exact directional Hessian
at the largest adjacent slope-change bracket midpoint. Central finite-difference
slopes use widths .02/.01/.005. Audit tolerances: atol2e-4+rtol.05. The two smallest
widths must agree with autograd and each other; all observations, including failed
checks, are kept. These are fixed-resolution audits, not guaranteed convergence
or full-spectrum Hessian measurements. Alpha-neighborhood curvature differences
may evaluate just outside [0,1]. A completed job is not a passed numerical audit.

## Timing analysis: exact information timeline

At the decision BEFORE update d (weights theta_(d-1)):
- Available geometry: the already completed update d-1, including its exact
  displacement projection and cosine.
- Available persistence: its observed validation-loss increase, mean4/mean8,
  EWMA and current A-validation loss.
- No outcome of update d or later, and no test loss, enters these scores.

Horizon0 predicts delta A_test at update d; horizon1 predicts d+1; horizon3 predicts
d+3; horizon7 predicts d+7. This is equivalent to geometry leads1/2/4/8. Targets
use fixed thresholds .02/.05/.10. No target classifier fitting, probability
calibration, or threshold tuning. AUROC/AP are reported; no Brier score is claimed
for uncalibrated raw scores. Early/middle/late thirds, paired circular block
bootstrap intervals with lengths8/16 and300 resamples, and temporal-shift placebos
are included. These are within-trajectory uncertainty diagnostics, not independent
seed inference or valid permutation p-values for a nonstationary process.

The original frozen OLMo predictor is also evaluated separately in
`original_frozen_prediction_metrics.json`, keeping its original feature timestamp
and leads0/1. Do not equate that older fitted loss-history feature set with the
new latest-completed-change baseline. Main timing summaries omit trajectories
failing the A-acquisition gate; exclusions are exported. Appendix runs are kept
separate and are not pooled into the primary forecast analysis.

Bundled compact records mean the old large ZIPs need not be uploaded again.
`existing_sources` can optionally include full run directories, SQLite files, or
share ZIPs. They are read-only; conflicting duplicate trajectories are refused.

## Answer-token control

For each seed, permute A..F once. A uses the same two labels in BOTH paired arms.
B uses four labels overlapping A in `counterbalanced_shared` and the complementary
four in `counterbalanced_disjoint`. Instructions explicitly describe each mapping.
Underlying examples, class targets, content truncation budget, batches, A labels
and A training are matched. Both arms use the already selected standard-task Adam
LR, with no relabel-specific tuning. The control is token competition plus codebook
change; it does not isolate token identity from every effect of instructions.
Compare the paired arms, not a disjoint arm to the unpermuted historical baseline.
JSON metadata records actual token IDs and answer-set overlap. Loss is decomposed
within each task's OWN answer set; leakage can include the other task's tokens.

## Appendix optimizer definitions (read before citing)

These are explicit single-device preliminary variants implemented here; no
external optimizer package or distributed process group is needed.

`shampoo_block`: hidden-layer2D matrices only,128x128 blocks, cumulative left/right
Gram accumulators, FP64 eigendecomposition of symmetrized accumulators, inverse
fourth roots with trace-relative ridge1e-4 plus absolute floor1e-12. Roots refresh
at step1 and every10 steps. Raw preconditioned blocks are assembled, then the whole
matrix direction is grafted to the current raw gradient's Frobenius norm. Classical
momentum .9 acts on that preconditioned/grafted direction. Nonhidden parameters
use Adam with LR=.02 x matrix LR, beta1=.9, beta2=.99. This is a blocked, SGD-grafted
Shampoo hybrid; NOT a performance claim about Distributed Shampoo or all variants.

`muon_fp32`: hidden-layer2D matrices only; EMA momentum .9 with Nesterov mixing,
five quintic Newton-Schulz iterations (3.4445,-4.7750,2.0315), FP32 arithmetic,
Frobenius normalization with epsilon1e-7, and sqrt(max(1,rows/cols)) matrix scaling.
Nonhidden parameters use Adam with LR=.001 x matrix LR. Original implementations
may use BF16 for the transform; this explicit FP32 variant facilitates auditing.
The transform is approximate and nonlinear: F(history+current) generally differs
from F(history)+F(current). Only total-displacement diagnostics and controlled
history resets are used. Optimizer update orthogonalization is not itself an
activation reference-frame rotation test.

Both hybrid parameter assignments exclude embeddings, heads, biases and vector
norm parameters from matrix optimization. Tied parameters are counted once.
Auxiliary LR ratios are fixed, not extensively tuned. One seed is a preliminary
appendix pilot, not independent confirmation of optimizer superiority.

Sources:
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- https://arxiv.org/abs/1802.09568
- https://github.com/KellerJordan/Muon
- https://arxiv.org/abs/2102.07686

## Files, storage, restart and validation

`REPORT.txt` and `summary.json`: all completed/main/calibration outcomes.
`calibration_selection.json`: complete candidate scores and acquisition eligibility.
`history_probes.json`, `path_audits.json`: immediate controls and numerical audits.
`acquisition_targets.json`: retention at fixed B-valid thresholds .75/1/1.5/2,
with one or four consecutive qualifying endpoints. Unreached targets are missing.
`timing_existing/`, `timing_new/`: timestamp-aligned predictions and uncertainty.
`original_frozen_prediction_metrics.json`: previous frozen baselines, separately.
`failures_and_skips.json`: no silent fallback or missing-result-as-success.
`results.sqlite`: all prompt-level observations and full path points.
`overview.png`: generated at queue completion; numbers are available earlier.

Only ONE active rolling checkpoint is kept, saved every8 updates; it includes
model and native optimizer state. Completed-job checkpoints are removed after
records commit. Model/data revisions, source hashes and deterministic minibatches
permit replay, but arbitrary additional raw-tensor analyses require recomputation.
No raw gradients or activation dumps are archived. The output cap defaults80GiB
with40GiB filesystem reserve. Atomic checkpoints temporarily need another copy.
HF download caches are outside the output cap; filesystem reserve still applies.

Stop is cooperative. Run status until alive=False, then Launch/resume. Interrupted
chunks replay from the last atomic checkpoint, replacing only their unsaved
records. Restart creates a NEW directory; it never deletes old research. Worker
locks prevent duplicate launch. Status refresh is read-only. Current code, frozen
records and environment versions are checked for consistency on resume.

Export creates `optimizer_generality_share.zip` via a consistent SQLite backup,
without model checkpoints or tokenized training corpora. Keep the original run
if you need restart/replay. Tests exercise tiny Llama/GPT-NeoX CPU models, exact
pause/resume, native channel sums, matrix optimizer state restoration, norm
matching, timing leakage, label pairing and calibration independence. See
VALIDATION.txt for the actual checks run; real GH200 performance is not promised.
