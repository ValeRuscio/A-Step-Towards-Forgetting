# Natural forgetting: optimizer components and finite-update coupling

Research question: Which components of an actual optimizer update accompany which kinds of forgetting, and how often does finite-update coupling materially change the outcome?

This package records native full-parameter Adam training. It does not reset momentum, optimize a retention method, fit a rotation, train a predictor, or change weights during the recorded trajectory. Diagnostic interpolation happens in a deterministic replay after the trajectory. It is restored to the exact endpoint before the next training step.

## Run

1. Extract the entire folder onto the GPU machine. Keep its .py files together.
2. Open `Run_Natural_Components.ipynb` in that folder, using the existing working PyTorch environment.
3. Run Setup, then Launch/resume. The worker runs in the background with output in `worker.log`.
4. Rerun Status whenever wanted. It does not launch, reset, or automatically poll.
5. Stop is cooperative. Wait for `alive: false`, then Launch/resume continues the same run.
6. Results prints the compact report and a plot. Export creates a ZIP with an SQLite snapshot, JSON summaries, provenance and code; no checkpoints.

No training weights or prior result archives need to be uploaded. Models and text datasets download from pinned revisions on first use. The notebook reuses existing settings when reopened, preventing accidental resets. Scientific settings/code are immutable within one output directory. Use the optional New run cell only for a genuinely new experiment; old results are preserved. Do not use it to resume.

The default interpreter is the current notebook interpreter. The notebook does not install or upgrade PyTorch. `requirements.txt` lists the small supporting dependencies; use a separate environment if installation would disrupt ongoing experiments.

## Default protocol and cost

- SmolLM2-135M and Pythia-410M; pretrained revisions pinned.
- Seeds 51, 52, 53 (fresh relative to previous supplied studies).
- Two answer conditions: counterbalanced shared and disjoint tokens; 12 trajectories total.
- A: SST-2 sentiment; B: AG News topic classification, 600 A updates and 128 B updates.
- Adam beta1=.9, beta2=.99, no weight decay, epsilon 1e-8. LR frozen at 8e-5 / 5e-6 from the previous study, not selected on these test outcomes.
- Global clipping at 1.0; deterministic batches and eager attention; model in eval mode to disable dropout, with gradients enabled.
- 1024 training examples per task, 32 validation examples, 128 test examples. Exact means/gradients refer to these fixed populations, NOT the full source datasets.
- Full A-validation confusion/leakage gradients at every B update. Test labels are evaluated but never used for training or selecting paths.
- Eight updates sampled uniformly without replacement from all 128, using a fixed seeded draw independent of outcomes. Up to one additional update per validation-defined stratum: A increase >.05, ordinary A change <=.01, both A and B increases >.05. Overlap is retained with multiple reasons, not counted as independent paths.
- Each selected update gets a full path on all 32 A-validation examples and the first 32 fixed A-test examples (selected by the dataset's seeded order, not loss).
- Thus 192–264 population paths are planned by default, depending on enrichment overlap/availability. 1536 natural B updates and 7200 A-acquisition updates.
- Paths start at 5 uniform points, refining to 9/17/33. All three loss components and individual channel terms must meet convergence tests. Extra finite-difference probes are saved outside that core grid.
- Per-example exact derivatives at alpha=0,.5,1 on a fixed subset of 8 examples per population. Per-example losses at all natural endpoints. All path points have layerwise population projections.

This is a large precision experiment, not a promise to finish in 24 hours. Every launch has a 23-hour session budget; Launch/resume continues after a budget pause. Real GH200 wall time depends heavily on second derivatives and refinement. Full-resolution runs can require multiple sessions. Each model/seed/condition finishes its natural trajectory and its selected paths before moving to the next, so partial data remain interpretable. Do not treat unfinished cases as negative findings.

For a smaller pilot, BEFORE the first launch set `SETTINGS['seeds']=[51]`. This reduces replication and is not a substitute for the full study. For more precision, before starting a new run increase `uniform_updates`, `path_max_points`, or `path_population` (the latter cannot exceed validation/test sizes). Do not tune these choices based on favorable test results.

## Mathematics and what is measured

For designated answer tokens S and true answer y:

    total = -log p(y)
    confusion = -log[p(y) / sum_{j in S} p(j)]
    leakage = -log sum_{j in S} p(j)
    total = confusion + leakage

All text targets are one-hot. Full-vocabulary and restricted-answer accuracy, answer mass, and class/example identities are retained. Loss improvement is not automatically improved discrimination.

Native Adam channels use the SAME updated denominator and bias-correction clock:

    h = -lr * beta1*m_prev / [(1-beta1^t)*(sqrt(v_new/(1-beta2^t)) + eps)]
    c = -lr * (1-beta1)*clipped_gradient / [same denominator]

The actual displacement d is the FP64 difference of stored FP32 endpoints. The numerical residual r=d-h-c is explicitly retained. The decomposition does not treat Adam as unpreconditioned SGD, nor reset its pretraining/task history. It uses history accrued during this study's A/B training; pretrained optimizer states are not provided.

For each loss component L, along theta(alpha)=theta_before+alpha*d:

    Delta L = g(0).d + integral_0^1 (1-alpha) d^T H(alpha) d d_alpha

Compute exact autograd HVPs for d,h,c and retain:

    hh, hc, ch, cc, hr, cr, rr
    d^T H d = hh + hc + ch + cc + 2hr + 2cr + rr

The mixed contribution is the weighted integral of hc+ch (not hc alone). The residual contribution is the weighted integral of 2hr+2cr+rr. Symmetry and decomposition closure are checked. Numerical sums use FP64; the primary model computation remains FP32.

This is an exact smooth-path identity, subject to numerical audits, not a novel theorem. The empirical contribution is the sizes, signs, replication, behavioral components, and coverage of the measured terms. A mixed Hessian term is a path-dependent mathematical attribution, not independently identified causal interaction and not proof of a reference frame. The interpolation is a straight segment between real endpoints, not a claim that the optimizer traversed all intermediate weights.

Every update stores pre-update component gradients projected on h,c,r,d and the clipped training gradient, a five-vector Gram matrix, layerwise projections, and recent changes in those projections. A changed projection does not by itself identify which individual vector rotated; the saved Gram matrices and norms characterize their joint geometry. No activation topology or covariance eigenspectrum is being substituted for loss derivatives.

## Numerical audits

- Adaptive Simpson slope integral vs observed endpoint change.
- Independently integrated Hessian vs finite remainder; individual curvature terms must converge, not just their cancellation.
- Channel slope-sum and Hessian decomposition closure; h/c mixed symmetry.
- At midpoint and at the location of largest total slope change: central finite differences for loss slope, total directional Hessian, and mixed h/c derivative; three epsilon values. The final two widths must pass and agree.
- Native FP32 endpoints exact; interpolated assignments round to FP32. Hessian checks detect when this approximation is unreliable.
- On an unresolved native path, a separate parameter-FP64 derivative recheck is recorded by default. It never upgrades a failed native audit or replaces native results. Model internals may still have FP32 operations; this is not a claim of fully FP64 kernels. It does not step or mutate optimizer state.
- Replay checks the exact stored weight hash after EVERY replay update before using a cached path. A discrepancy stops the case rather than silently using different training.
- No graph or parameter-sized gradient arrays are written to disk.

`fully_audited` requires component integration and derivative checks. `capped` refers only to unresolved integration at the resolution cap: an uncapped result can still have failed derivative checks. Inspect both. The derivative checks sample locations; they do not certify every point or every example.

## Scientific denominators and interpretation

Uniform-path prevalence is reported per trajectory and population, with planned, completed, audited, unresolved and class counts. Enriched events are not pooled into prevalence. Do not discard unresolved paths from the denominator silently. Eight uniform samples per trajectory still give limited precision for rare mechanisms; raw denominators are supplied rather than misleading prompt-level confidence intervals.

Natural-update scalar summaries cover all 128 updates. Path attribution uses a fixed 32-example test subset, while endpoint functional summaries use 128 test examples. Do not compare their means as if they were the same population.

An acquisition gate requires A-validation NLL gain >=.1 and restricted accuracy >=.6. Failed gates remain visible, but primary acquired-task inference excludes them. Those thresholds are protocol choices, not proof of full task mastery. Seeds are replication units; examples and updates are dependent. Answer conditions are paired within seed, not independent extra replications.

The shared/disjoint pair has identical A codebook, training schedule, text/class identities, and (audited) A-anchor weights. B answer-token identities and the instruction codebook change; token frequencies/embeddings are not identical. This tests sensitivity to answer-codebook overlap, not an isolated effect of overlap independent of every token property.

A mixed term is labeled 'material' when its absolute value is >=.01 nats AND >=25% of the absolute observed component loss change. These thresholds are fixed descriptive choices; raw values are retained for sensitivity analysis. Large means >.05 nats. Classification is mutually exclusive: unresolved; ordinary/nonlarge; favorable-to-harmful reversal; nonlinear amplification; first-order dominated; other. It does not mean a mixed term is beneficial/harmful in isolation when other terms cancel it.

## Files to send back

- `REPORT.txt`, `summary.json`
- `natural_summary.json`, `event_components.json`
- `path_summary.json` (integrals, levels, FD audits, optional FP64 checks)
- `uniform_prevalence.json`, `answer_condition_pairs.json`
- `settings.json`, `environment.json`, `code_hashes.json`
- The SQLite snapshot in the share ZIP includes all per-update/per-example records, path points, Hessian points and prompt derivatives. `path_spec` contains population identities and endpoint hashes.

## Storage and restart behavior

Only one case's A-anchor and rolling checkpoint are retained; an atomic temporary checkpoint briefly adds a third checkpoint-sized file. Completed-case checkpoints are removed AFTER the completion record is committed; observations and settings are retained. `minimum_free_gib=40`, `max_output_gib=80` prevent uncontrolled output growth. Downloaded Hugging Face cache is external to this output cap and is not deleted automatically.

The exported ZIP refuses to run when no results database exists. It uses a consistent SQLite backup even while training runs. The current output directory is printed in Setup and returned on launch. Never search for a restart ZIP while your completed results are in the original directory.

Checks covered by `test_components.py` and `test_integration.py`: known polynomial Hessian/mixed/residual terms, gradient additivity, exact path integrals, cached endpoint mismatch, validation-only selection, pause inside an unfinished path and resume, tiny Llama/GPT-NeoX end-to-end replay, paired A anchors, idempotent rerun, empty-export refusal. Full pretrained CUDA runs must still pass their own recorded audits.
