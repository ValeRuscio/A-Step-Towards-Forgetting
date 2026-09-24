# Precise natural forgetting: matched populations and finite-update paths

This is a focused follow-up to the completed natural-geometry run. It separates the two recorded prompt forms from entity identities, uses the exact gradient for each evaluated prompt, and compares each layer immediately before and after a natural optimizer update. It does not train an intervention branch or modify the original research files.

## Run tonight

1. Extract every file into a new directory on the GH200.
2. Open `Run_Precise_Forgetting.ipynb` using the existing working Python/CUDA environment.
3. Run Setup, Settings, then Start. Default input: `/home/ubuntu/1/runs/olmo_association_v1`.
4. Refresh Status manually. The notebook does not stream training output. Keep the remote instance running; the browser can disconnect.
5. Run Results for printed numbers/plots; Export creates `precise_forgetting_share.zip`. An export is also created automatically when the worker finishes, pauses, or fails.

The default session is 11.5 hours, with 10 minutes reserved for reporting. Timing is a session limit, not a guarantee that every numerical check resolves. A long derivative or export may overrun that reserve. Launch again to continue completed-prompt caches; there is no need to restart from scratch.

`stop_run()` requests a cooperative stop. Wait for `alive=False`; run Start to resume. `fresh_run()` creates a new sibling output directory. Changing scientific settings also creates a sibling automatically. Time and disk budgets can be changed on resume. Prior research outputs are never erased.

## Default queue

The following retrospective event selections are prioritized:

- Seed 1: updates 21, 48, 20, 47.
- Seed 2: updates 11, 35, 96, 10.
- Seed 3: updates 7, 121, 6, 8.

The actual queue interleaves seeds. Events outside a source's configured B horizon, or missing seeds, are omitted by defaults. These selections were informed by the previous results: they diagnose known behavior and do not constitute independent confirmation of a predictive hypothesis.

Pass 1 measures both natural endpoints at every selected event, including full-population exact gradients and every layer's geometry. Later passes give all events three path points, then refine unresolved events to five, nine and at most seventeen points. At least nine points are required before declaring a default path resolved. Resolved paths stop early; unresolved paths retain explicit flags. More points can be requested before the initial launch. Completion of the queue does not imply that every numerical convergence check passed.

The nearest original saved fork is loaded for each event, then the original B minibatches are replayed to the selected update. Full before/after parameter hashes are checked when continuing an existing event. The existing source files and pinned model revision are preserved. No new full model or optimizer checkpoint is written by this runner, so resume replays a short original prefix instead of accumulating large state files.

## Exact population matching

All original entities (normally 32) are evaluated in A-valid, A-test, B-valid and B-test. Each entity has one recorded prompt form in valid and one in test, with the same target within a task. Each prompt has its own full-parameter, unclipped loss gradient. Population results are means of those same prompts' gradients and losses. There is no small monitor proxy and no gradient sketch.

The paired table separates:

- prompt-form main effects;
- entity main effects;
- prompt-by-entity interactions;

for observed loss changes, linear predictions and finite remainders. The balanced variance components add to the total variance. They are descriptive for these two fixed templates, not causal attribution or a random-effects estimate over all possible prompts. Test prompts used in retrospective diagnosis are not being claimed as untouched confirmation data.

## Tracking the nonlinear effect

Let theta0 be the natural pre-update weights, theta1 the actual Adam output and Delta=theta1-theta0. We evaluate the diagnostic straight line theta(alpha)=theta0+alpha*Delta. Training never proceeds from an intermediate point, and a finally block restores theta1 after the probes, including on a cooperative stop.

For each exact entity/prompt loss:

- `loss_change = L(theta1)-L(theta0)`;
- `initial_slope = gradient L(theta0) dot Delta`;
- `exact_finite_remainder = loss_change-initial_slope`;
- path losses and exact autograd directional derivatives at every stored fraction;
- sampled slope-sign reversal brackets;
- Simpson quadrature of the directional derivative, checked against the actual endpoint difference and a nested coarser grid.

The endpoint remainder requires no Hessian approximation. “Exact” here means the actual saved losses and autograd derivative of the FP32 model, subject to floating-point error. Interpolation is rounded to the model's FP32 parameters. Small central finite differences at fractions 0, 0.5 and 1 use two widths to audit selected entities in every split. Endpoint audits extend slightly outside [0,1] for the symmetric derivative check. Audits are sampled, not exhaustive. A failed audit is retained rather than silently discarded.

Native Adam supplies history and current-gradient numerator channels h and c under the same updated denominator. Because the actual FP32 displacement is not perfectly h+c, the runner also records the roundoff residual Delta-h-c. It integrates each channel's projection along the same path:

    integral gradient L(theta(alpha)) dot h d alpha
    integral gradient L(theta(alpha)) dot c d alpha
    integral gradient L(theta(alpha)) dot (Delta-h-c) d alpha

Their sum equals the integrated total slope to numerical precision. Comparing each integral with its initial projection shows how its contribution changes over the finite displacement. These are path-dependent diagnostic contributions, not “what would happen if we removed momentum” and not unique causal shares.

This follow-up does not claim to separately measure all hᵀHh, hᵀHc and cᵀHc terms. It directly measures the finite remainder and the derivative along the actual combined displacement. Slope changes can contain all of those second-order contributions and higher-order effects. That avoids comparing a two-entity Hessian estimate to a different eight-entity loss, as in the previous run.

Default integration criterion for every entity/prompt: both endpoint closure error and difference from the coarser grid must be below `integration_atol + integration_rtol * max(abs(loss_change), abs(integral))`. All sampled derivative audits must pass for `resolved=True`. These numerical checks are not rigorous bounds on an arbitrary function, and a sampled grid can miss narrow features or extra sign reversals. Report unresolved cases as unresolved.

## Layer-specific, recent geometry

Every layer's residual output and the final normalized hidden state are captured at both endpoints. There is no averaging of layer signatures in the primary saved data, and the reference is the immediately preceding state, not the distant A anchor.

For each prompt and layer, the runner computes activation displacement, old/current loss sensitivity projected onto that displacement, and per-token old-sensitivity projections. Unlike the earlier last-token-only local projections, these contractions include every token position of this exact prompt. They remain local derivatives: do not sum activation-site projections across layers, which would double count effects. Separately recorded parameter-layer gradient projections do sum over disjoint parameter groups.

Last-token geometry includes mean shifts, covariance spectra/effective ranks, principal subspace angles, and held-out orthogonal-alignment errors. The descriptive orthogonal fit can include reflections; no fitted transformation is applied to the model. Entity identities used to fit alignment and the task semimetric are disjoint from the topology/evaluation landmarks.

H0/H1, identity-labelled MST edges and neighbor retention are measured on fixed landmarks before and after the update. Both Euclidean distance and a frozen pre-update margin-gradient semimetric are recorded. This semimetric is not the full Fisher or loss Hessian. Distance scales are fixed within each event; normalized lifetimes should not be compared across unrelated events as absolute distances. Global rotation or translation alone preserves Euclidean PH.

This focused run does not repeat the earlier broad random-matrix scans. The previous results retain those measurements. The priority here is matched derivatives, finite-step behavior, prompt/entity separation and layer-specific changes.

## Output and storage

- `REPORT.txt`, `summary.csv`, `matched_gradients.png`: population-matched loss/linear/remainder comparisons.
- `within_update_paths.png`: A-valid versus A-test loss and derivative along each measured path.
- `measurements.sqlite`: per-entity path probes (including channel and parameter-layer projections) and layer geometry.
- `seed1_update021/update.json` (and corresponding event folders): source provenance, full before/after hashes, actual channel norms and per-matrix update statistics.
- Event folders: `prompt_entity.json`, `path_summary.json`, `finite_difference.json`, and compressed last-token NPZ arrays.
- `code/`, `provenance.json`, `settings.json`, `hardware.json`: reproducibility records.
- `precise_forgetting_share.zip`: consistent SQLite snapshot and readable results, excluding raw NPZ arrays.

All-token activations/gradients are temporarily saved under this runner's event `scratch/` folders so individual prompts can resume. After an event/split's layer measurements are committed, only that reconstructible scratch is removed. Per-token projections and compact last-token arrays remain. Source checkpoints and previous experiment outputs are never cleanup targets. The package does not retain every raw all-token tensor permanently or promise lossless retention of all possible analyses.

Defaults reserve 25 GiB free and pause when output exceeds 30 GiB. Storage checks can overshoot by one prompt/file. There are no new 12-GiB rolling model checkpoints. GPU memory contains model/optimizer state, three full displacement vectors, parameter gradients and the current graph. GH200 capacity is intended, but full-model memory/time were not locally benchmarked.

## Validation

Run `python test_precise.py` for offline tiny original-OLMo numerical tests, and `python test_lifecycle.py` for the detached worker controls. The numerical tests verify:

- unchanged natural post-update weights and optimizer state;
- mean single-prompt projections agree with an independently computed full-population gradient;
- paired variance partition and known polynomial quadrature;
- finite-difference derivative checks and endpoint integration closure;
- summed channel integrals agree with the total integrated derivative;
- interrupted interpolation restores the natural post-update state;
- resume fills missing cached probes without duplicates;
- report/plots/export and completed-run resume.

Dependencies are listed in requirements.txt. Reuse the existing CUDA torch installation; do not reinstall torch blindly. Tests ran locally with torch 2.8.0 / transformers 4.57.6 on CPU. Full OLMo-1B/GH200 execution is not claimed as locally validated.
