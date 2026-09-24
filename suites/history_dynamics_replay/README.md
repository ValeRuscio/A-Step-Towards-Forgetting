# History dynamics: magnitude, moving gradients, and the age of momentum

A separate, read-only-source replay study for `natural_forgetting_components_v1`.
All Python code is included. No results ZIP is required: point the notebook at the original GPU run directory containing settings.json, results.sqlite, environment.json, and data/.

## Start

Extract this folder separately from the previous experiment; open Run_History_Dynamics.ipynb. Set SOURCE to the original run folder (default /home/ubuntu/9/runs/natural_forgetting_components_v1). Run Setup, then Launch/resume. Status is manually refreshed; logs do not stream into the notebook. Stop is cooperative. Launch/resume continues saved work; Restart creates a different run and is not used to resume.

Use the same Python environment, model revisions, and hardware as the source. Torch/Transformers/NumPy versions are checked. Every replayed endpoint must match the source's exact weight hash. A mismatch stops the analysis instead of silently treating a different run as replication. The public output-head getter compatibility fix is included.

The worker waits while the original worker lock is held. If the original worker starts again while replay computes, replay pauses at its next check to avoid sustained GPU contention. It does not control, stop or edit the original worker. Do not run unrelated GPU experiments concurrently unless you intentionally manage resources outside this package.

## Snapshot scope

When the original worker becomes idle, the replay freezes a consistent SQLite snapshot and copies the saved example populations. Only completed natural trajectories at that moment enter the queue. An original run paused halfway can therefore yield a partial snapshot. This is printed in queue.json and status, not mislabeled as all 12 cases.

Later source completions are not silently added to an existing snapshot. For a full 12-case analysis, wait for the original run to complete before launching this notebook. For an interim analysis, launch on the available cases; a later full snapshot needs a new output directory. The package never writes the original database or changes training settings.

## What is calculated

Let g_t be the component gradient immediately before B update t, and u_t either the native history displacement h_t or current-gradient displacement c_t. Both use Adam's actual updated second-moment denominator. The current batch gradient is already available at this timestamp; this is retrospective mechanism analysis, not a forecasting claim.

1. Magnitude and orientation, globally and for every parameter block:

    P_t = g_t . u_t = ||g_t|| ||u_t|| cos(theta_t)

   Save both norms, cosine, signed projection and consecutive-vector cosines. Near-zero norms give null cosines, never fabricated angles. Decompose the change in P using a symmetric product rule: first split norm-product vs cosine, then split the norm-product into its two factors. All three signed contributions sum to Delta P. This hierarchical convention is exact but not the only possible allocation for a three-factor product.

2. Whose vector changes:

    Delta P = ((g_t+g_prev)/2).(u_t-u_prev)
            + (g_t-g_prev).((u_t+u_prev)/2)

   These are channel-displacement-change and task-sensitivity-change contributions. The first B update has no fabricated preceding B vector; cross-time fields begin at update 2. The midpoint rule here is an algebraic identity, not a numerical approximation. Gradient change includes norm and orientation; use the first decomposition to distinguish them.

3. Ages of optimizer history:

   - task_A: all of the native first moment at the end of A training, subsequently decayed;
   - B_age_1_4: preceding B gradients aged 1–4 updates;
   - B_age_5_16: preceding B gradients aged 5–16;
   - B_age_17plus: all older B gradients.

   Ages refer to the displacement about to occur: the previous update has age 1. All gradients are the actual globally clipped training gradients. Each age bin is projected after applying the SAME updated denominator, learning rate, beta1 factor and bias correction as the native history channel. They are not independently renormalized.

   The reconstructed numerator is checked against the native FP32 exp_avg. Roundoff is retained explicitly; age projections plus the roundoff projection recover the native history projection. Age bins are exact in real arithmetic and numerically audited, not approximate exponential windows with discarded tails. Task-A contribution is not genuine pretraining optimizer history: the source began from pretrained weights with a fresh optimizer.

4. Adaptive scaling vs numerator changes:

    u*_t = K_t D_t N_t
    K_t = -lr/(1-beta1^T)
    D_t = 1/(sqrt(v_new/(1-beta2^T)) + eps)
    N_t = beta1*m_prev for history; (1-beta1)*g_train for current

   Between updates, split Delta u* into changes in N and changes in K*D using the symmetric product rule. Then split Delta(K*D) into adaptive inverse-denominator and scalar clock contributions. Project these vector changes on the average component gradient. Their sum, plus native implementation-roundoff change, equals the channel-change contribution in item 2. The learning rate is fixed in this source, so the clock term is bias correction.

These are parallel/nested decompositions, not independent quantities to add indiscriminately. For example, orientation_effect and gradient_change answer different questions; adding them would double count. No denominator replacement or moment deletion is performed in training. These are attributions along the reconstructed native trajectory, not causal estimates of alternative optimizer runs.

## Populations and behavioral outcomes

By default measure exact confusion/leakage gradients on every saved A-valid example (32 in the source) and the first 32 saved A-test examples in their fixed source order. Total gradients are their algebraic sum. test_population can be increased BEFORE launch to cover all 128 held-out examples, with proportional derivative cost. All gradients are full population means for the named subset, not minibatch proxies or Hessian surrogates.

The matched-population pre/post changes are computed at each update. Full original validation/test changes are saved separately and explicitly labeled. Do not conflate a 32-example derivative population with the source's 128-example test mean. Per-example identities are retained in spec records.

Confusion and leakage retain the original exact identity: -log p(y) = -log[p(y)/sum_answer p] - log sum_answer p. Pre/post restricted and full accuracy are recorded on the replay population.

Event windows use ORIGINAL A-validation total increases >.05 nats, not held-out outcomes. The full time series is retained; windows may overlap, and updates/layers/examples are not independent replications. No prompt-level significance claims are generated. Acquisition-failed cases remain flagged. Filter acquired=true and complete=true for primary summaries. Signed sums and sums of absolute contributions are supplied separately. Cross-time signed sums telescope and must not be presented as measures of total variation.

## Storage, runtime, resume

The original package deleted completed checkpoints. Consequently this replay reconstructs task A from its pinned model/batches, then verifies the A anchor and every B endpoint. It does not repeat finite-path HVPs or curvature grids.

Only the current case's A acquisition checkpoint/anchor is saved. The last 16 clipped B gradients, older-bin accumulators, previous component gradients and channel vectors are held in HOST RAM, not accumulated as disk dumps. On resume, the saved A anchor is loaded and B updates are replayed to rebuild this state; completed diagnostic records are reused. Only the last completed pre-update gradients must be reconstructed before collecting new cross-time records. Completed case checkpoints are removed only after their completion record is committed.

This trades some repeat compute for much smaller disk usage. It is intended for GH200's large host memory; Pythia can use tens of GiB for the gradient ring alone and roughly 100 GiB or more of host RAM overall. It is not intended for a small-RAM laptop. Native training/derivatives use FP32 model parameters and FP64 reductions, with eager attention and dropout disabled, matching the source.

Session budget defaults to 23 hours, free-disk reserve 40 GiB, output cap 60 GiB. These are not runtime guarantees. Source cache files are external to the cap and never deleted. Resume starts a fresh session budget. A replay hash mismatch must be investigated, not overridden by raising a tolerance.

## Outputs

- REPORT.txt: compact signed age summaries.
- summary.json and replay_audits.json: completed denominators, exact endpoint checks, age residuals and algebra checks.
- geometry_summary.json: per-trajectory/per-layer signed and absolute sums, cosine means and available counts.
- age_summary.json: per-trajectory age contributions, including roundoff.
- global_timeline.json: every measured pre-update global geometry and matched outcome.
- event_windows.json: validation-selected event-aligned records, explicitly dependent.
- results.sqlite: full update records, including all layers and all audits.
- overview.png: projections and cosines over time for the latest available case.
- snapshot_manifest.json, source_settings.json, source_environment.json, source_code_hashes.json: provenance.
- history_dynamics_share.zip: snapshot of replay results and scalar summaries, without model checkpoints, gradient histories or original source SQLite.

Inspect age residual audits before interpreting age-bin splits. Algebra closure validates bookkeeping, not the assumption that a directional attribution is a causal intervention. Model/seed summaries must retain variation rather than pooling every update into an apparently large sample.

## Validation

`python -m unittest discover -p 'test_*.py' -v`
Tests include explicit-gradient age reconstruction beyond age 16; native Adam moment agreement; exact cross-time, magnitude/orientation, adaptive-scaling and age projection identities; undefined zero-norm angles; tiny Llama and GPT-NeoX source generation and endpoint-exact replay; interruption/resume with preserved records; source database preservation; export and analysis. No full pretrained GH200 replay has been executed locally.
