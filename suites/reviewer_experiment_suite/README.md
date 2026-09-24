# Reviewer experiment suite

Full Python code and a quiet Jupyter wrapper for four additional experiments.
It does not edit or resume the older research runs. Use a new output directory.

## Run

1. Extract the entire ZIP together on the GH200 server. Open `Run_Reviewer_Experiments.ipynb`.
2. Use a Python environment with the existing working CUDA PyTorch installation.
   Qwen3 requires a compatible Transformers version; this release was tested with
   Transformers 4.57.6. Your `/home/ubuntu/4/env_qwen3/bin/python` is a suitable
   candidate **if its CUDA import still works**. The notebook checks it.
3. Run setup, settings, and Launch/resume once. Refresh only the status cell to
   check progress. Results/plots are displayed only when explicitly requested.
4. Stop requests a cooperative pause. Launch/resume continues the same run.
   Restart returns settings for a **new directory**, keeping the old run intact.
   Scientific setting changes require that new directory; hours and storage
   limits can change on resume.

`SETTINGS = study.defaults(...)` enables all four experiments, two models
(OLMo-1B and Pythia-410M), three **new** seeds (11/12/13), and two task families.
That is **216 conditions**, not a promise of completion in one night.
A session runs for up to 11.5 hours and then pauses; an active tensor operation
must finish before it can stop. Begin with the smaller notebook overnight
preset, which runs all four experiments on Pythia-410M, seed 11, both tasks
(**36 conditions**). Replicate with seeds 12/13 in a separate run. Do not treat
one seed as conclusive. The notebook leaves this choice visible.

No automatic installation changes an active environment. `requirements.txt`
lists additional packages; reuse your working CUDA Torch rather than replacing
it. Package installation, if needed, belongs in a separate environment.

## What is implemented

### 1. Prediction-triggered interventions

At pre-update t, the alarm uses measurements from **pre-update t−1** and earlier.
No current B minibatch gradient, post-update t outcome, or test-set loss enters
that decision. A direct, unclassified projection score and a frozen loss-history
classifier each trigger a one-step learning-rate multiplier of 0.25. The first
nine updates are a fixed warm-up; at most 16 interventions occur per 128 updates.
Cutoffs are the source 80th percentile, frozen before these new runs.

Each policy has a random schedule with exactly the same number of interventions,
drawn independently of loss outcomes from the same eligible steps 10..128.
The random branch starts from the same anchor and uses identical B minibatches.
These random schedules are generated after the policy supplies its count; they
are controls for intervention frequency, not independently calibrated online
policies. All branches use unchanged nominal LR on nonintervention steps.

Frozen source-only prediction comparisons:
- Raw signed projection (no classifier; AUROC/AP only).
- Strong loss history: losses, log step, lags and rolling changes at 1/2/4/8 steps.
- Norm-only, cosine-only, norms + cosine, projection-only logistic ablations.
- Loss history + projection versus history + old/new gradient interference.
- Population gradient interference and raw clipped B-minibatch interference.
- Leads 0 and 1; large-loss thresholds 0.02, 0.05, and 0.10 nats.

Lead 0 is a current-gradient-informed diagnostic, not an advance warning.
The direct projection is g_A · actual FP32 Adam displacement on new natural runs.
Historical records provide g_A · (history + current), omitting their recorded
roundoff residual. The bundled original OLMo source used A_test/B_test gradients;
new policies deliberately use validation gradients. This measurement shift is
recorded and must be disclosed. We do not silently refit to new trajectories.

Frozen baseline coefficients and the compact 384-update original OLMo source
are bundled with database provenance. These additional baselines were designed
after inspection of the original project results. Reanalysis of already-seen
models is therefore retrospective; the fresh seeds are the independent test.
There is no 0.05 singular-value cutoff: 0.05 here is a loss-event threshold.

### 2. Optimizer-history and LR controls

All conditions branch from one shared A-trained model **and optimizer state**.
The small factorial uses LR multipliers 0.25, 1, 2 and preserved, reset-first,
and reset-both moments. Reset moments keep Adam's step counter. The learning
rate changes apply throughout B training, and the original 1× preserved
condition is shared with the natural trajectory.

A separate short branch study compares preserve/reset-first/reset-second/
reset-both/fresh optimizer. Its first actual displacement is rescaled to the
preserved-history step's global Euclidean norm, audited after FP32 assignment.
Subsequent updates use the nominal learning rate. This tests first-step
direction at matched global size; it does **not** match every layer's norm,
future step sizes, or optimizer trajectories. A fresh optimizer also resets
the clock and is reported separately.

Report both A retention and B acquisition, equal-step and first-crossing of
a shared B-validation-loss target. Final, mean and peak loss and mean positive
excess are saved; a better final endpoint does not erase an earlier spike.
The B target is the completed natural branch's final validation loss. Conditions
that never reach it are reported as missing, never extrapolated.

### 3. Independent frame-intervention replication

Fresh seeds; layers 0 and 1 fixed in advance. Analyze fixed update 20 plus the
first two updates with calibration-entity A-validation loss increase >0.05.
Held-out frame-evaluation identities do not enter that event-selection mean. Do not select layers,
events, or rotations using test rescues. The fixed ordinary step need not be
nonforgetting; it is a time-selected comparison, not a guaranteed negative.

Fit a proper, uncentered orthogonal Procrustes map from post- to pre-update
activations using validation calibration prompts. Registry calibration and
held-out evaluation use disjoint entity IDs; all IDs were taught during task
training. Natural-text calibration and evaluation use disjoint examples.
Use the same fitted map at the four history/current corners and alpha=1.
Full-update corners include the measured floating-point residual only at 11.
The intervention patches the last-token residual output of the selected block.

Controls: sham, inverse rotation, mean shift, scale, four random-subspace
orthogonal maps sharing the fitted rotation eigenangles, and four random
additive displacements matched per prompt and per corner to the rotation's
activation-displacement norm. Orthogonal controls match eigenangles, not patch
norms. The same maps and per-entity random vectors are used across the factorial.

Save endpoint change, baseline patch effect, change in forgetting relative to
the correspondingly patched pre-update state, and change in the factorial
interaction. A reduction in interaction alone is not a rescue. Include all
selected cases and all controls, including failures. Evaluation patches are
immediate interventions, not demonstrations of sustained training repair.

Selected natural events also get exact autograd path slopes, nested Simpson
integration with local error estimates, channel integrals, finite-difference
slope checks and directional-Hessian/finite-difference curvature audits near
the largest sampled slope change. Resolution is bounded at 65 points. Unresolved
paths are explicitly flagged; endpoint closure alone is not sufficient. The
curvature location and path grid may miss a narrower feature; no full Hessian
spectrum or exhaustive path certification is claimed.

Centered activation spectra, distance-filtration H0/H1 (at most 32 held-out
points), and held-out MSE to pre-update activations are also saved. PH uses a
common pre-update distance scale across corners. Exact rotations preserve
Euclidean distances/PH; a functional rescue need not change these quantities.
This is not a random-matrix-theory or gradient-spectrum experiment.

### 4. Different task family

The new family is sentiment classification on **SST-2**, followed by four-class
news-topic classification on **AG News**, using the language model's native
full-vocabulary next-token output. It is natural-text sequential supervised
fine-tuning, not open-ended generation or a general benchmark of all LLM skills.

Datasets are pinned to commits at first launch, and exact tokenized examples,
labels, split IDs, source indices and truncation flags are saved. SST-2's public
validation split is held out as the test population because its test labels
are hidden. Separate disjoint training examples supply validation/calibration.
AG News uses its public test split. Exact-text and tokenized-prefix duplicates
are rejected across the sampled splits. Train/validation/test defaults are
1024/32/128 per task. Answers A/B or A/B/C/D must be distinct single tokens; the
suite refuses unsupported tokenizers instead of changing targets silently.

Content is truncated to fit 256 tokens while keeping instruction and answer
cue. There is no instruction-model chat template; all models receive the same
plain classification prompt. One-hot targets use xlogy so 0 log 0 is exactly zero.

Dataset sources:
- https://huggingface.co/datasets/stanfordnlp/sst2
- https://huggingface.co/datasets/fancyzhx/ag_news

## Exact answer decomposition

For designated answer set S, M=sum_{j in S} p_j, and targets q supported on S:

    KL(q || p_full) = KL(q || p_S / M) - log M
                       confusion          leakage

For one-hot targets this is the same decomposition of NLL. The two components
can move in opposite directions. Restricted-answer accuracy and full-vocabulary
argmax accuracy are both recorded. Every new evaluation and every frame corner
contains entity/example rows; the decomposition is not inferred from accuracy.
`analyze_existing.py` also extracts it from old `support=-log M` measurements.
When an immediately preceding mass measurement is absent, the change is marked
unavailable rather than assumed to be zero.

## Storage, interruption, and scope

- SQLite scalar/per-example records, compact spectra/PH; no raw gradient arrays.
- One shared A anchor per case; one rolling checkpoint for the active condition.
- Atomic replacement needs space for the old and new rolling checkpoint.
- A completed case retains its A anchor for follow-up; no automatic deletion
  of prior research, Hugging Face caches, or completed anchors.
- Default output cap 180 GiB and filesystem reserve 40 GiB; resume after changing
  those operational limits if needed. A checkpoint write reserves additional
  estimated model/optimizer storage. These are safeguards, not exact quotas.
- Checkpoint every 8 updates; a stop can replay at most that unfinished chunk.
  Completed frame-condition/path records are reused during deterministic replay.
- Gradients are exact over the saved validation population on natural trajectories;
  the projection policy needs only g_A. The first norm-matched step also records its A projection. Other branches avoid
  those extra backward passes. Test gradients are used only for selected offline path/curvature audits.
- The output is **not lossless archival of every tensor**. It preserves the
  declared measurements, pinned inputs, training recipe and restart state.
  Other tensor analyses require replay. This avoids the earlier 200+ GiB arrays.
- Bitwise replay is checked on CPU in tests, not guaranteed across hardware or
  library versions. Record and keep the environment manifest.

Use `study.export(SETTINGS)` to produce a consistent, checkpoint-free share ZIP.
Use `study.show_results(SETTINGS)` for numbers and A/B trajectory plots.
`python analyze_existing.py path/to/four_vector_share.zip --output reanalysis`
runs the extra prediction baselines and answer decomposition without training.

## Interpretation

Three seeds give limited uncertainty estimates. Report seed-level effects and
all selected frame cases; do not treat hundreds of updates as independent
replications. New-task progress, acquisition quality and the numerical audits
must accompany forgetting claims. A favorable initial projection becoming a
harmful finite step is a possible route, not assumed to be the dominant one.
