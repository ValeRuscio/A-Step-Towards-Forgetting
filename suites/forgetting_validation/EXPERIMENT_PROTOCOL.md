# From finite-update geometry to consequential forgetting

## Evidence already available (2026-09-17)
The completed OLMo four-vector archive permits a direct retrospective test of persistence and entity breadth. It does not require another GPU run. See results/REPORT.txt and selected_events.json. All cutoffs below are exploratory choices for these already inspected runs; freeze them before new-model evaluation.

Seed 1/update 21 changes old-task KL from 0.089042 to 0.376914. All 32 entity losses rise by more than 1e-4 nats; the median rise is 0.275463. Two-answer majority-label accuracy drops from 100% to 50%; mean full-vocabulary probability of that majority answer falls from 0.738248 to 0.549216. No saved endpoint from step 21 through 128 regains the pre-event loss (minimum 0.285584). The final loss is 0.844272 and two-answer accuracy is 34.375%. This is evidence for broad and lasting deterioration along the observed continuation. Subsequent B updates remain possible contributors; no counterfactual cause is identified.

Seed 1/update 48 and seed 3/update 7 also worsen all 32 losses, but do not lower two-answer accuracy immediately. They return to within 1e-4 of their respective pre-event losses for eight consecutive endpoints starting 6 and 7 additional updates later. These are transient worsening episodes relative to already-degraded pre-event baselines, not proof of restoration to the A anchor.

## Aim 1 — Persistence with meaningful baselines
For every update, retain A/B losses, correct-label probabilities, restricted answer accuracy, full-vocabulary top-1 accuracy, and entity-specific outcomes. Save after each update and at the A anchor. Compare (a) the immediately preceding state, (b) the A anchor, and (c) an unmodified continuation in branching experiments. Save absolute metrics, not only ratios.

Primary observation window: 32 subsequent updates; descriptive secondary windows: 8 and 16. Recovery: eight consecutive endpoints within 1e-4 nats of pre-event loss. Report censoring explicitly near run end. Also report mean signed excess, mean positive excess, peak, and the fraction of observed endpoints above baseline. Later loss cannot be attributed to a single event merely because the curve stays high.

For an event-specific causal comparison, fork from the exact pre-event model and optimizer state: natural update, omitted update, and shortened actual displacement (e.g. 0.25, 0.5, 0.75). After the initial branch step, restore the SAME natural post-event optimizer state in every branch and use the same subsequent minibatches for 32 steps. This isolates the effect of changing that parameter displacement conditional on a common optimizer state; it is not equivalent to native lower-learning-rate training or a fresh optimizer. Include a separate fully native lower-learning-rate branch if that policy is of interest. Report B acquisition cost alongside A retention. Omission alone is not a learning-matched control.

## Aim 2 — Functional breadth and prompt/entity separation
Evaluate multiple fixed, token-distinct templates for each entity at the anchor, before and after each selected event, and at +8/+16/+32 steps. Keep new paraphrases disjoint from training and from model/event selection. Save entity, template, target distribution, full-vocabulary target log probabilities, answer mass, margin, top-1 token correctness and KL. The existing archive supports two-answer accuracy and majority-answer probabilities, but NOT full-vocabulary top-1 decisions from those scalars alone.

Report loss and decision changes per entity, per template, and their interaction. For inference, resample entities with all their templates together; seeds/models remain separate replication units. Do not treat multiple paraphrases or updates as independent model runs. The controlled association setup teaches the same entity identities; unseen-entity generalization is a different experiment requiring learnable structure in the labels.

Large mean KL and reduced accuracy are different outcomes. Correct-label probability and log loss can deteriorate without crossing a decision boundary. Conversely, average recovery can conceal a subset that remains damaged.

## Aim 3 — Geometry tied to loss, then mechanism tests
### Natural measurements
Keep the current multimodel run intact. At every update it already measures history h, current channel c, full A/B gradients and their temporal changes. Test advance prediction on held-out seeds and models using frozen validation-defined event thresholds, with lag-one features as a separate analysis from imminent-minibatch features. Compare against pre-update A/B loss, prior loss change, update norm, gradient norm, and gA dot delta. No new predictive claim until evaluated.

At selected large A_valid increases and matched ordinary A_valid updates, collect the following on the SAME fixed entities and prompt templates at t-1, t, t+1 and selected within-step fractions:
- Every layer's final-token residual activation x_l and local old-task sensitivity dL/dx_l. State that the local derivative holds other sites fixed.
- Recent changes relative to the preceding state, separately from total anchor drift.
- Activation norms, centered covariance eigenvalues and effective rank; pairwise-distance changes; fitted orthogonal alignment residuals; task-sensitivity alignment; full parameter-layer gradient/channel projections.
- Full-population path slopes and adaptive curvature audits from the existing code. Endpoint remainders remain usable even when their within-path attribution fails convergence.

Fit any reference-frame map on calibration entities only, then assess it on held-out identities and templates. Choose layers and hyperparameters on validation data. A good Procrustes fit by itself is not evidence of a loss-relevant frame mismatch.

### Topology and spectral controls
Compute Vietoris–Rips H0/H1 persistence on the SAME entity-template points, with explicit site, centering, normalization, distance, scale, and filtration cutoff. Record empty diagrams, distance to diagonal, and point count rather than reporting only median H1 bottleneck. Avoid conclusions from a small empty or numerically degenerate H1 diagram.

Compare Euclidean geometry to task-weighted geometry (e.g. projected logits or a fixed, explicitly defined sensitivity metric). A common rigid rotation preserves Euclidean persistent homology. A task-weighted metric can change, but then its changes must be separated from changes of the point cloud itself. Hold the metric fixed within one comparison and repeat with the alternative state's metric as a sensitivity analysis.

Nulls: (1) a common orthogonal rotation as an invariance sanity check; (2) centered spectrum-preserving row mixing, with the all-ones direction fixed, as a covariance-spectrum control; (3) target-label permutations for task association; (4) matched ordinary updates. Row mixing preserves selected spectrum but not entity identities; label association is precisely what the comparison disrupts. Do not call all such nulls RMT or infer a universal Marchenko–Pastur threshold for correlated residual activations. Prefer calibrated finite-sample null distributions with recorded dimensions and aspect ratios.

### Direct tests of the reference-frame hypothesis
At a prespecified layer/site in a damaged state, apply a calibration-fitted orthogonal map toward the preceding reference frame ONLY at evaluation. Compare unpatched, sham, mean-only, scale-only, fitted rotation, random orthogonal maps matched in eigenangles, and additive controls matched in activation displacement magnitude. Measure A and B on held-out entities/templates. Also test the fitted map at an ordinary event and the inverse-direction map where well-defined.

A rescue must improve absolute A outcomes, not just reduce a factorial interaction. Orthogonal patches preserve activation norms, but can still be strong generic perturbations; random and non-rotation controls are essential. Simultaneous A/B improvement is particularly informative, though not by itself proof of recovering stored knowledge. If the effect appears only under a persistent evaluation hook, call it a decoding/interface rescue, not durable training-time retention.

### Direct tests of history/current interaction
At the same pre-event checkpoint, evaluate the parameter plane theta(u,v)=theta_before+u*h+v*c, keeping the native updated Adam denominator fixed. Include the measured residual r explicitly or audit the endpoint mismatch. At equal u=v=alpha compute the finite factorial contrast L(alpha,alpha)-L(alpha,0)-L(0,alpha)+L(0,0). This tests interaction in these chosen directions; it is neither the entire finite remainder nor a universal causal decomposition of Adam.

Compare natural displacement, history-only and current-only displacement with norm/step-length controls and B-loss-matched comparisons over a validation-selected grid. Hold model/optimizer state fixed for loss probes. If continuing branches, document the optimizer-state convention described under Aim 1. Resetting Adam's first moment changes more than a single abstract 'memory angle' and is not a clean rotation test.

## Staged compute plan
1. Now, CPU: run the supplied retrospective analyzer; no interruption to current GPU work.
2. Finish existing numerical audits and model replication; review before scheduling more expensive work.
3. Targeted follow-up: two validation-selected events per completed model/seed (large increase and ordinary control), one fixed prompt/entity panel, 32-step continuation; save scalar tables and compact layer matrices only.
4. Run activation-frame/topology measurements and the small intervention panel at selected layers. Expand to more layers or points only if a predefined audit/effect justifies it.

No raw full gradients per prompt, no unbounded checkpoint history, no duplication of old experiment arrays. Retain the required branch-start checkpoint/state and pinned source revision. Use an explicit storage cap and resumable per-unit records.

## What would support or weaken the hypothesis?
Support: reproducible broad and persistent impairment, a frame/sensitivity change that precedes it beyond loss/norm baselines, held-out selective rescue by the fitted map beyond matched controls, and comparable behavior across models. This would support a mechanism in the tested setting, not a proof of a universal cause.

Weakening evidence: magnitude terms account for the effect without frame-specific rescue; random rotations work equally well; topology changes occur just as strongly in ordinary updates; the apparent predictor fails held-out seeds; or most loss rises resolve promptly. These are useful outcomes and should remain reportable.
