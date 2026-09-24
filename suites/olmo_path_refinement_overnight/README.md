# Targeted refinement: seed 1 updates 20–21, seed 2 updates 10–11

This follow-up copies and reuses the completed precise-forgetting measurements. It never edits the previous results or source checkpoints. It adds adaptive path points, shrinking-width derivative audits, and interior layer geometry for the four unresolved natural updates.

## Start

1. Extract the complete package on your GH200, keeping all Python files together.
2. Open `Run_Path_Refinement.ipynb` in your existing working environment.
3. Run Setup, Settings, Start. Default previous results: `/home/ubuntu/4/runs/olmo_precise_forgetting_v1`.
4. Refresh Status manually. The detached worker can continue while your browser is disconnected, provided the instance stays running.

The previous directory must contain settings.json, measurements.sqlite and the four event folders with update.json, finite_difference.json and path_summary.json. An extracted `precise_forgetting_share.zip` also supplies these files, provided the original checkpoint source path in its settings still exists on the server.

Default session: 11.5 hours, with 10 minutes reserved for finishing/export. Long indivisible operations can overrun the reserve. There is no guarantee that every prompt will converge in that time. The queue order is seed 1 step 21, seed 2 step 11, seed 2 step 10, seed 1 step 20.

Run Start again to resume. Already-running launches are harmless. Call `stop_run()` in a new cell to pause, then refresh until alive=False. New scientific settings create a sibling output directory; previous results remain intact. Change time/disk budgets in place when resuming. No new full model/optimizer checkpoints are written; interrupted prompt calculations are resumed from cached measurements after reconstructing the natural event from original checkpoints.

## What is refined

Every original entity and all four A/B valid/test populations remain separate. Previously converged per-entity quadrature is reused only if its recorded errors satisfy the current tolerances. Previously unresolved prompts are prioritized.

For each unresolved entity/prompt:

- Start with eight intervals covering the whole actual update, using the existing samples.
- Evaluate extra midpoint/quarter-point derivatives where needed.
- Compare fine and coarse Simpson estimates AND compare each interval's integrated derivative with the actual loss difference across that interval.
- Split the interval with the largest normalized discrepancy.
- Stop when local and global checks pass, or a point/depth cap is reached.

Default caps are 257 cached points per prompt and interval depth 12. Cap-reached results stay explicitly unresolved. The local absolute tolerance is proportional to interval width; the relative tolerance uses local loss change/integral magnitude. These are numerical tests, not rigorous quadrature bounds. Very narrow structures or FP32 noise can still prevent reliable resolution.

The full parameter gradient belongs to that exact prompt and target, not a monitor proxy. Native history/current channels use the original shared Adam denominator. Their path integrals plus the floating-point residual channel reproduce the total integrated directional derivative to numerical precision. These are contributions along the chosen actual-displacement path, not predictions of what would happen after resetting momentum.

## Derivative checks

For every prompt, audit the stored point with the largest sampled absolute directional derivative. Also re-audit any points that failed the previous run's finite-difference test. Widths shrink from 0.005 down to 0.00015625 by halves.

A check passes only when two successive finite-difference estimates agree with each other and with autograd under the configured tolerances. The estimates and errors are saved, including failures. Symmetric endpoint checks extend slightly outside the update interval. A failure is not automatically evidence of a model phase transition: finite-difference truncation, FP32 resolution and implementation issues must remain possible explanations.

The per-prompt summary distinguishes quadrature_passed, cap_reached and resolved. Resolved requires both quadrature and derivative audits. A finished queue may contain unresolved prompts.

## Geometry inside the sharp regions

After refining the four jobs, select up to two interior fractions per event. Selection uses the most frequently occurring per-entity sampled slope peaks among A-valid/A-test, with a minimum separation of 0.025. This is retrospective diagnostic selection, not held-out prediction. Some events may have fewer than two interior peaks.

Capture every layer at 0, the selected interior fractions, and 1 for all A-valid and A-test entities. Compare successive selected fractions, retaining layer-specific:

- all-token activation/sensitivity projections and per-token contributions;
- last-token covariance/subspace geometry and held-out orthogonal alignment;
- Euclidean and frozen local task-semimetric H0/H1, MST edges and neighbor retention.

The reference for each geometry pair is the previous selected fraction. Its task metric and distance scale are fixed within that pair; normalized values are not automatically comparable across pairs. Local activation projections from different layers must not be summed as a loss attribution. Full B-valid/B-test paths are refined, but additional interior geometry defaults to A only.

Intermediate parameters are diagnostic evaluations. The actual post-update weights are restored afterward, including cooperative interruption of probes. Training never continues from a virtual intermediate point.

## Outputs

- REPORT.txt and refinement_audits.csv: completed prompt counts, resolution/cap flags and integration errors.
- refined_paths.png: individual worst-endpoint-change A-valid/A-test examples, explicitly labelled by entity. Different entities have different adaptive grids, so the figure does not average mismatched samples.
- measurements.sqlite: copied original probe/endpoint data for the four selected events, plus all additional probes. Table `refined` contains per-entity adaptive leaves, channel integrals and finite-difference attempts. Table `slice_geometry` contains intermediate layer comparisons.
- Event `slices/` directories: compact last-token NPZ arrays for the selected geometry pairs.
- path_refinement_share.zip: all stored scalar/per-entity/layer records via a consistent SQLite snapshot, plus summaries/plots. Raw NPZ tensors are excluded.

Temporary all-token tensors are generated under this follow-up's full_token_scratch/scratch directories, reused through hard links, and removed after the corresponding measurements are committed. Only this runner's reconstructible scratch is cleaned. Original experiment arrays/checkpoints are untouched. These cleanup operations do not claim to retain every possible raw-tensor analysis.

Defaults reserve 25 GiB free and cap the output directory at 30 GiB (checked between jobs). Temporary-file checks are conservative and may count hard links more than once; the free-space reserve is also checked at progress boundaries. A single operation can temporarily exceed a soft limit.

## Verification and limits

`python test_refinement.py` runs an offline test with a sharply localized analytic derivative, explicit non-convergence caps, channel-sum checks and a tiny original-OLMo end-to-end run. It checks input-database immutability, derivative audits, intermediate geometry, resume and export. The underlying precise runner had already passed natural-weight/optimizer restoration tests.

Tests run on CPU with torch 2.8.0 / transformers 4.57.6. The full GH200 workload is not benchmarked locally. Reuse your working CUDA torch installation rather than replacing it. None of these retrospective diagnostics establish a universal cause of forgetting by themselves.
