# History, persistence and finite-update geometry

Extract this ZIP into a **new folder**. Keep all files together and open
`Run_History_Geometry.ipynb`. Run setup, settings, and Launch/resume. The worker runs
quietly in the background; rerun Status whenever you want an update. No previous
research directory or checkpoint is changed. No previous ZIP upload is required.

## Default study and why these models

SmolLM2-135M gives a relatively economical Llama-family test; Pythia-410M provides
replication in GPT-NeoX at a larger size. Both already have measured acquisition
settings, avoiding a new Qwen environment or a new tuning campaign.

1. SmolLM2 Adam, seeds **41/42/43**.
2. Pythia-410M Adam, seeds **41/42/43**.
3. SmolLM2 momentum SGD, the same fresh seeds, to test history without Adam's
   adaptive denominator. This last tier can be disabled **before first launch**
   with `SETTINGS['include_smollm_momentum'] = False`.

Nine natural trajectories in total. Each learns A for600 updates, then B for128.
Before fixed B updates **8,32,64**, three branches start from the identical model
AND native optimizer state:

- `preserve`: native history.
- `reset`: zero the first-moment/momentum buffer once.
- `reset_blocks`: the same reset, then rescale that first displacement to match
  the preserved displacement's norm in every transformer block (one additional
  group contains embedding/readout/other parameters).

Each branch runs **32 total updates including the intervention**, i.e.31 further
updates. Same minibatches, same scalar LR, same clock and same second moment at
branch initialization. Subsequent updates use the native optimizer without any
further resetting or norm matching. The norm match is an explicit first-step
intervention, equivalent to group-specific displacement rescaling at that step;
it is not ordinary unchanged-LR training at that one step. Its optimizer state is
what the reset branch naturally computed; moments are not rescaled to pretend
that this was a native update.

Default: **27 fixed forks,81 continued branches**,1152 measured natural B updates,
2592 measured branch updates,5400 A-training updates. Fixed path subset: seed41,
forks8/32, all three conditions, across all three model/optimizer combinations:
**18 paths**. Checkpoints are not selected from future test damage. Failed A
acquisition does not trigger retuning or cherry-picking replacement seeds.

Every pair isolates the specified history intervention at one common state.
Across-optimizer comparisons still have different learned A anchors; do not
interpret those comparisons as a clean beta1-only experiment.

## Time budget, storage, stop/resume

Default session budget: **23 hours**. A GH200 wall time has not been benchmarked.
The full queue may require another launch. Pause is cooperative between batches
and operations; a long kernel or checkpoint write must return first.

- Launch/resume is idempotent while a worker is active.
- Stop requests a cooperative pause. Wait for `alive=False` before resuming.
- A time-budget pause needs only Launch/resume again.
- Restart creates a new output directory; it never deletes prior work.
- Scientific configuration changes require a new directory. Hours, interpreter
  path, threads, checkpoint interval and disk budgets can change on resume; an
  incompatible software-version change is refused.

Only **two rolling model/optimizer checkpoints** are used: natural state and
active branch state. Paths save scalar records, not tensor snapshots. Intermediate
raw gradients/activations are not retained. The default output cap is60GiB, with
40GiB filesystem reserve; HF caches are outside the output cap. Atomic checkpoint
replacement temporarily needs another copy. In-memory copies fit much more
comfortably in GH200 unified memory than on a small-RAM machine.

Records beyond the last durable checkpoint are deterministically replayed on
resume. Completed path points are reused; completed branches are skipped. The
natural state is restored after every set of branches. The preserved branch's
first-step losses must exactly reproduce the corresponding natural update.

## Task and numerical settings

Unchanged SST-2 sentiment A -> AG News topic B, fixed source dataset revisions.
Each task:1024 training,32 validation,128 test texts; max prefix256 tokens.
Full-vocabulary one-hot cross entropy, with answer-set confusion/leakage and
restricted/full-vocabulary accuracy recorded separately. The natural text-hash
confirmation pool is separate from the earlier LR-calibration pool. Fresh seeds
change data sampling and minibatches; they do not guarantee disjoint text examples
across seeds or from the previous inspected run. They are held-out trajectories,
not a claim of completely unseen language corpora or pretraining examples.

Pinned model revisions match the previous run. FP32 parameters, FP64 loss and
scalar reductions, no autocast/TF32/flash attention, dropout inactive, eager
attention, deterministic operations, zero weight decay, clipping norm1,
microbatch2 x accumulation2. LRs fixed from the prior run before these new seeds:
SmolLM2 Adam8e-5; Pythia Adam5e-6; SmolLM2 momentum SGD.01. Adam beta1=.9,
beta2=.99, epsilon1e-8; classical SGD momentum.9, no Nesterov/dampening.

Acquisition gate: A-validation improvement>=.1 nat and restricted accuracy>=.6.
All runs and gate failures are retained. Primary prediction excludes failed
acquisition. Branch tables explicitly flag failures, rather than presenting
failure to learn as good retention.

## Experiment1: functional consequences of a one-time history change

Every B update, including every branch update, records full validation/test
populations, exact full32-prompt A-validation gradient, actual assigned displacement,
projection, cosine, norms, native history/current projections and channel residual,
layerwise total projections/norms, training-gradient interference and clipping.
A Gram table uses A-valid gradient, clipped B-minibatch gradient, history displacement,
and current displacement. This is explicitly NOT the old full-A/full-B-population
four-vector definition. A full B-population gradient is not needed for these tests.

Tables compare immediate loss, final loss, mean excess, mean positive excess,
peak excess, accuracy, confusion/leakage, and loss-increment lag-one correlation.
Report both A retention and B acquisition. Fixed B-validation loss thresholds
.5/.75/1/1.5 require four consecutive endpoints; compare A at the end of that
qualifying interval. Unreached targets are missing, not favorable results. These
thresholds are a secondary acquisition-matched description, not exhaustive matching.

Paired effects share an anchor fingerprint. Seed-level summaries average forks
within each trajectory; forks/prompts are not independent seeds. Three fresh
seeds remain a small replication sample. No automatic significance claims.

## Experiment2: frozen forecasts beyond loss persistence

`development_records.json.gz` contains compact records from the previous
optimizer-generality study: standard trajectories that passed A acquisition,
seeds21/22/23. Those already-inspected outcomes are now explicitly DEVELOPMENT
labels. Their origin hash and included jobs are recorded. No new outcome is used
for fitting. `frozen_predictors.json` is already fitted and bundled.

All models use the same ridge logistic class, C=.1, no class balancing, train-only
standardization, damped Newton solver with normalized-gradient tolerance1e-8.
Same source rows within a scope, no per-feature-set tuning. Feature sets:

- Strong loss history: current A/B validation loss, means/stds of last1/2/4/8
  changes, extrema, fixed EWMA, update index.
- Geometry: preceding projection/cosine/update norm/old-gradient norm,
  history/current projections, training interference, projection change.
- Combined; secondary loss+projection and loss+channels ablations.
- Raw projection and latest validation increase, with fixed orientation.

Decision BEFORE d: all features stop at completed update d-1. No current-update
geometry or current/future test loss enters a feature. Horizon0 predicts d;
horizons1/3/7 predict d+1/d+3/d+7. Event thresholds .02/.05/.10 nats; primary .05.
AUROC/AP for all scores; Brier/log loss/calibration bins only for fitted probability
predictors. Never interpret raw dot products as calibrated probabilities.

Primary source scope is pooled old data. Secondary source-specific fits distinguish
within-model fresh-seed replication, cross-model transfer, and cross-optimizer
transfer. The source scope is exported; do not mix them in one headline number.

Quiet-before-decision subset: no >threshold A-test increase in the previous four
completed updates. Recent-harm subset is its complement. These labels are
retrospective outcome stratification ONLY, not observable validation-only trigger
policies or predictor features. For horizon0, positive events in the quiet subset
are onsets under this predeclared definition. Later horizons are quiet-at-decision
forecasts, not necessarily first onsets. Missing positives/negatives yield null
AUROC/AP. No subset-specific refitting. Paired combined-minus-loss differences
are exported per trajectory; report seed variation, not pooled prompt confidence.

The stronger baseline may win. A null incremental-geometry result is valid and
must not lead to sign flipping or new target-seed tuning.

## Experiment3: independently audited finite-update curvature

At each scheduled path, interpolate the actual first-step displacement:

    theta(a)=theta0+a*Delta
    Delta L = g(theta0).Delta + integral_0^1 (1-a)*Delta^T H(theta(a))*Delta da

All terms use the SAME A_valid population and token losses. Loss/slopes are
adaptively refined on nested Simpson grids9->17->33->65. Independently compute
exact autograd directional Hessians on5->9->17 points and integrate their weighted
values. This is a directional curvature measurement, not a full Hessian spectrum.
The finite remainder computed by subtraction is never relabelled as an audited
Hessian integral.

At the largest adjacent slope-change bracket midpoint, central finite differences
at widths.02/.01/.005 audit both slope and Hessian. The two smallest widths must
pass and agree; these checks may evaluate slightly outside [0,1]. Tolerance:
atol2e-4 + rtol.05 times the relevant magnitude. Nested Simpson error estimates
are diagnostics, not rigorous error bounds. Actual-versus-interpolated endpoint
loss roundoff is recorded. Capped/unresolved paths remain explicitly unresolved.

Paired path tables separate intervention effects into initial-projection and
finite-remainder differences and report the independent weighted-Hessian result.
The Hessian interpretation is supported only where both paired paths pass closure,
curvature integral and derivative audits. Unsupported numerical kernels fail
explicitly rather than silently switching precision or claiming curvature.

## Files you will get

- `REPORT.txt`: readable completed results (available during the run).
- `results.sqlite`: all prompt-level and scalar diagnostic/path records.
- `natural_summary.json`, `branch_summary.json`, `paired_branch_effects.json`.
- `seed_level_branch_effects.json`: dependent forks reduced within each seed.
- `prediction_metrics.json`, `paired_prediction_differences.json`, prediction rows.
- `path_audits.json`, `paired_path_decomposition.json`.
- `overview.png`: A-versus-B paired final effects, by fork and by trajectory.
- `history_geometry_share.zip`: exported summaries, consistent SQLite backup,
  and code; excludes model checkpoints and tokenized training data.

Keep the original output folder for resume. Scalar archives support the stated
analyses, but arbitrary later activation/weight analyses require deterministic
replay. Checkpoint compression/quantization is not used.

## Environment and tests

Use the existing CUDA-enabled GH200 Python. The notebook checks imports and
selects `/home/ubuntu/ml_env/bin/python` if appropriate. It does NOT upgrade Torch
or change any environment automatically. `requirements.txt` lists dependencies.
No Qwen3 or new optimizer implementation is required.

Run `python -m unittest -v test_followup` for small CPU Llama/GPT-NeoX tests, exact
pause/resume, state preservation, norm matching, no-future-feature leakage, and
quadratic path identities. See `VALIDATION.txt` for actual performed checks.
Real-model GH200 numerical convergence, task acquisition, and runtime remain
empirical outcomes; tiny-model tests do not establish them.
