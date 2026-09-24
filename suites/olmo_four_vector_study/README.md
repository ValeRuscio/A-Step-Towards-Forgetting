# Four-vector natural forgetting study

Run the supplied notebook with the same working Python/CUDA environment as your previous OLMo study. Keep all supplied Python files together. Set `SOURCE` to the **original** `olmo_association_v1` directory containing `config.json`, `seed*/anchor.pt`, `seed*/data.json`, and any original `fork*/fork.pt` files. A share ZIP containing only measurements cannot replace these checkpoints.

The notebook remembers its last output directory and saved settings across kernel restarts. The notebook launches a detached worker and stays quiet. Re-run **Refresh** whenever you want progress; use **Show results** for numbers and plots. **Stop** requests a cooperative stop. Once `alive=False`, the launch cell resumes saved work. No fresh experiment is needed after a normal pause. Changing scientific settings creates a separate directory automatically, preserving old results. `restart(SETTINGS)` explicitly creates a fresh directory after the worker stops.

## What is measured

The four vectors at natural update t are:

- `history` (h): the old first moment's contribution to the upcoming Adam displacement, using the **updated** denominator and original learning rate/bias correction.
- `current` (c): the clipped current **training minibatch** gradient's contribution, under that same denominator.
- `gA`: the unclipped gradient of the complete fixed A evaluation population's mean loss.
- `gB`: the unclipped gradient of the complete fixed B evaluation population's mean loss. It is distinct from the training minibatch gradient used in c.

The defaults use all original `A_test` and `B_test` prompts. They preserve the original full-vocabulary conditional KL objective, targets, clipping, Adam, minibatch order, weights, and optimizer transition policy. No projection sketch, sampled monitor gradient, reset branch, or activation patch is used.

For every update 1–128 (or all available updates if fewer), each configured seed gets:

1. The complete 4×4 Gram matrix, all six pairwise dot products/cosines, and four norms, globally and separately for every transformer parameter layer. Embedding/readout parameters form a separate group. These are gradient/update measurements, **not activation covariance spectra** or a per-example gradient covariance spectrum.
2. Each vector's change norm and cosine to its previous pre-update version, globally and per layer.
3. The exact algebraic decomposition of every pair's dot-product change:

   Δ(u·v) = Δu·v_previous + u_previous·Δv + Δu·Δv.

   Small floating-point closure errors are recorded. This distinguishes movement of the update channels from movement of the task gradients.
4. A further separation of history movement into stored-momentum change, effective Adam-scale change, and their joint term. The scale includes the second moment, bias correction and learning rate; it is not solely the second moment.
5. Actual before/after task losses and the finite-update remainder:

   ΔL_A = g_A(0)·δ + R_A, where δ is the actual observed parameter displacement.

   Per-entity losses are retained for every evaluation split. The task gradients themselves are population means. This does **not** estimate a separate gradient for each entity or separate prompt/entity variance; the previous precise per-prompt experiment remains complementary.

Here h+c approximates δ up to floating-point rounding. The code records/projectively accounts for δ−h−c instead of silently replacing the actual displacement. To observe δ exactly, it forms the original optimizer step, evaluates diagnostics at restored pre-update/interpolated weights, and restores the actual post-update weights. Diagnostic gradients are cleared and optimizer state is not stepped during probes. CPU tests verify the resulting natural weights and optimizer state exactly against an uninstrumented run.

## Inside a finite update

For the configured diagnostic events, θ(α)=θ_before+αδ, 0≤α≤1. h, c and δ are held fixed to that actual update. gA(α) and gB(α) are recomputed across **all parameters and all prompts in the chosen population**. Both global and per-layer four-vector geometry are saved at each node.

The path loss derivative is L′(α)=g(α)·δ. Adaptive Simpson integration checks local endpoint-loss closure and nested-grid convergence. It separately checks the history, current, rounding, magnitude, orientation and joint contributions, so cancellation of two inaccurate contributions cannot alone pass the component checks. Finite-difference audits use the same complete population at α=0, 0.5 and 1, with two widths. Endpoint differences are one-sided.

Let D=||δ||, n(α)=||g(α)||, q(α)=cos(g(α),δ). Then:

R = D∫[(n−n0)q0 + n0(q−q0) + (n−n0)(q−q0)] dα.

The three terms quantify magnitude change, orientation change, and their joint contribution along this specified path. They are reference-dependent algebraic accounting, not separate causal effects. Their sum is checked against the finite loss remainder through the integral closure. A change in this angle is not proof of an activation reference-frame rotation.

**Resolution applies to population means and the specified component checks, not to each entity.** Per-entity losses are saved at nodes, but opposing individual derivative errors can be hidden by population averaging. A cap is never declared convergence. `PATH_DONE` means the configured computation finished, even if it reached a cap; inspect `quadrature_passed`, `cap_reached`, `slope_audits_passed`, and the leaf tables. Local quadrature tests are numerical diagnostics, not rigorous guarantees that no narrow feature was missed. FP32 interpolation and finite-difference rounding are retained and audited.

## Curvature at difficult locations

For each completed diagnostic path, the default selects the midpoint of the sampled interval with the largest absolute change in the A or B directional derivative per unit α. This is retrospective selection of locally rapid derivative variation; it need not find the global maximum or every difficult individual prompt. Increase `curvature_points_per_event` to inspect more such locations.

At each selected point, full-population autodiff Hessian-vector products measure hᵀHh, hᵀHc, cᵀHh, cᵀHc and δᵀHδ for both tasks. The contractions are also decomposed by the outer parameter-layer block, retaining cross-layer Hessian couplings; these are not within-layer Hessians. Mixed symmetry is audited, and δᵀHδ is independently compared with a finite difference of full-population directional gradients. These are local Hessian contractions, not Hessian eigenvalue spectra, Fisher matrices, or persistent homology. They test whether curvature along the actual motion is large, and whether history/current mixed curvature contributes; local contractions alone do not integrate the entire remainder.

This focused experiment adds no new PH computation. Euclidean PH does not detect a rigid rotation. Existing topology measurements should be compared to these loss-linked measurements at matched events and populations, without assuming that a topological change is required for forgetting.

## Overnight scheduling and storage

Defaults: 11.5-hour session, 10-minute reserve, 55% of the available session for dense natural trajectories, 30% for finite paths, and the remaining time for selected curvature. Seeds are processed in round-robin chunks of eight updates. The four initial path events are seed1/update021, seed1/update048, seed2/update011 and seed3/update007, when available. They are retrospective diagnostic cases from the earlier analysis, not new independent confirmation.

Every completed update, path node and curvature population commits to SQLite. On resume, the original checkpoints reconstruct the required state; this intentionally trades some replay computation for avoiding large continuation checkpoints. No raw gradients, activation arrays or new model checkpoints are written. The share ZIP contains **all saved measurements and the code**, but cannot reconstruct full tensors without the original source checkpoints. Default output guard is 15 GiB, and minimum free disk is 20 GiB. Do not delete the original source or required cached model revision.

Full 1B-parameter gradients, previous vectors and CPU copies require substantial host RAM (allow roughly 100 GiB free; actual use depends on model/checkpoint size) as well as GPU memory. GH200 is the intended environment; no GH200 benchmark has been run here. The code uses FP32 original training and disables fast attention kernels through the existing engine to support second derivatives. Start with the existing working CUDA environment; do not replace its torch build blindly.

The entire queue is **not promised to finish in one night**. Fixed session stage budgets ensure that dense trajectories do not consume the entire night before path work begins. Long atomic model operations can slightly overrun a deadline. If paused, increase session hours or run the launch cell again. Incomplete work is not a negative scientific result. Changing runtime budgets or chunk size does not invalidate existing records; changes to scientific settings get a new output directory.

## Files and inspection

- `four_vector_study.py`: new measurement, scheduling, reporting and lifecycle functions.
- `run_four_vector_study.ipynb`: quiet wrapper.
- `test_four_vector.py`: CPU tiny-OLMo integration checks, no downloads.
- Other Python files: bundled existing model/data, numerical integration and helper implementations. They must remain alongside the main file; no old notebook is needed.

In your configured results directory:

- `REPORT.txt`, `trajectory_summary.csv`, `trajectory.png`, `finite_paths.png`.
- `measurements.sqlite`: `trajectory` (seed/step), `path` (job/α), `curvature` (job/α/split). Each `record` field is JSON with full saved details, including layer-specific changes and per-entity losses.
- `seed*/path_accounting.json`, `slope_audits.json`, `curvature_points.json`, and source-event checks in `update.json`.
- `settings.json`, `hardware.json`, `provenance.json`, `worker.log`, `status.json`.
- `four_vector_share.zip`, created at pause/completion or explicitly through Export.

Read the richer records without printing everything:

```python
import sqlite3, json
with sqlite3.connect(str(OUTPUT / 'measurements.sqlite')) as db:
    row = db.execute('SELECT record FROM trajectory WHERE seed=? AND step=?', (1,21)).fetchone()
    event = json.loads(row[0]) if row else None
# Inspect on demand:
# event['temporal_changes']['vector_changes']['gA']['layers']
# event['temporal_changes']['layer_changes']['0']['history__gA']
# event['functional_consequence']
```

The strongest observational outcome would be a repeatable, layer-local change in these vectors that precedes forgetting, a resolved finite-path accounting of a substantial loss increase, and matching curvature at the difficult regions. This can sharpen a mechanism hypothesis. It does not establish generality across LLMs or causal necessity/sufficiency without additional evidence.
