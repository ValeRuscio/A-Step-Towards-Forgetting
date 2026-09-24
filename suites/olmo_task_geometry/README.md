# Task-aware geometry during forgetting

The original study is extended in the same two Python files. The notebook keeps the original background Run / restart, Stop, Refresh status and Show results workflow. No separate runner module or model-training framework is needed.

## Run

Extract the entire ZIP into a new folder on the GPU machine and open `Run_structure_study.ipynb` in the existing OLMo environment. Run from the top. SOURCE is discovered at the original association campaign (including `/home/ubuntu/1/runs/olmo_association_v1`); set SOURCE if it is elsewhere. Keep `structure_study.py` and `forgetting_mechanisms.py` beside the notebook.

The default runs the **new trajectory experiment only**, for seed 1 and actual B updates 20, 21, 46, 47. It records the states immediately before and after those updates. It does not infer the intervening trajectory from distant endpoints. Change `SETTINGS["trajectory"]["steps"]` for denser measurements. All steps must be within the saved source B schedule, with a preceding saved fork.

All 32 source entities are included for each of A_valid, A_test, B_valid and B_test. There is no first-four-entities panel. The same known entities occur in both prompt forms; this tests prompt-form transfer, not generalization to unseen arbitrary associations. The provided PH implementation supports 4–32 entities.

Default sites are the first, middle and last decoder-block outputs plus the final hidden representation. They are sampled at the final token, the position producing the answer. `layers=[]` selects them automatically; explicit indices can be supplied. Original weight, attention, RMT and NTK analyses remain available by setting `run_checkpoint_study=True`. Their methods are in CHECKPOINT_METHODS.md. The notebook sets that optional NTK panel to all available entities.

Run / restart stops the process currently tracked by SETTINGS and starts fresh in a new folder, preserving prior results. Refresh and Show results use SETTINGS's actual output folder. A kernel restart loses that in-memory association; if an old run is active, retain its results path to stop it. `study.launch(SETTINGS)` remains available to resume a compatible stopped run instead of starting fresh. Completed trajectory updates are reused; replay through skipped updates reconstructs the original optimizer state. Code or settings changes require a separate output, selected automatically when inactive.

## Measurements

### Actual training motion

The original model, Adam state, training minibatches, clipping, learning rate, and attention backend are replayed unchanged from the saved fork. Before each measured update, the full A_test mean gradient is computed without clipping. After the actual B training step, the code measures:

- the true parameter displacement;
- its inner product and cosine with the old A-test gradient;
- the observed A/B loss changes;
- the difference between observed A change and its first-order prediction;
- native Adam history/current-numerator contributions using the same updated denominator and bias correction, plus a displacement-reconstruction audit.

Positive `A_linear` means a first-order increase in A loss. History/current terms are not norm-matched, and the updated denominator depends on the current gradient. This is algebraic accounting of an actual update, not a counterfactual experiment with a different optimizer. Their contributions need not equal observed loss changes because of nonlinear effects and numerical rounding.

### Per-entity directional estimates

For each A/B test entity, symmetric finite differences along the actual parameter displacement are evaluated about the pre-update point at two spacings (0.05 and 0.025 of that displacement). The post-update weights are restored from saved CPU tensors even if a probe fails; optimizer state is not changed by probes.

The average A finite difference is checked against the exact full-panel gradient dot displacement. Per-entity spacing disagreement is also saved. FP32 rounding can make very small directional estimates unreliable. A successful average audit does not guarantee each entity's estimate; inspect both diagnostics. These derivatives are local estimates, not observed full-step loss changes.

### Task-sensitive hidden geometry

For each entity at each site, the code saves its final-position representation and derivatives of:

- `m = z_answer0 - z_answer1`;
- `t = logsumexp(answer logits) - logsumexp(other logits)`.

Let `c=sigmoid(m)` and `s=sigmoid(t)`. The coarsened output probabilities are `(s*c, s*(1-c), 1-s)` for answer0, answer1, and other. The full task KL to `(q0,q1,0)` is exact in these coordinates. Its coordinate gradient is `(c-q0, s-1)`.

The Fisher weights for prediction change in this **three-outcome distribution** are `(s*c*(1-c), s*(1-s))`. Together with the hidden Jacobians they define a positive-semidefinite, generally degenerate local metric. It does not distinguish redistributions within the other-token group. It is not the full-vocabulary Fisher matrix and is not the target-loss Hessian.

A frozen metric is obtained by averaging the A-anchor sensitivities across **all A_valid entities**. Its factor is applied to representations from every later state. This creates distances in a fixed projected feature space, calibrated without A_test prompts. A single A-anchor A_valid distance scale is retained at every state for each site/metric. This is a fixed reference construction, not a geodesic computed from a varying metric.

At earlier blocks, derivatives with respect to the final-position output hold the other token positions fixed. They cannot account for all changes propagated through the network. The local loss-gradient dot hidden displacement and Fisher-weighted length are diagnostics, not full causal loss attributions.

At the final hidden representation, the code also records angle and norm relative to the answer0-minus-answer1 decoder vector, and the exact margin decomposition:

`delta_margin = d_before @ delta_h + delta_d @ h_before + delta_d @ delta_h`.

Numerical reconstruction errors are retained. A change in angle does not automatically identify a rotation of an entire representation space.

### Labelled topology

For every task and prompt form, all entities participate in H0/H1 persistence under ordinary Euclidean and frozen task-sensitive distances. No independent per-state scaling is applied. The code also retains entity identities and computes:

- the filtration distance at which each entity's connected component first contains an opposite target class;
- nearest-opposite minus nearest-same target distance (excluding the entity itself);
- distances between validation/test forms of the same entity and nearest validation-entity identities;
- label-permutation controls preserving target-class counts, with the same random permutations across states;
- H0/H1 bottleneck changes across the actual update.

Labels annotate the geometry; they do not change the unlabelled PH computation. A pure rotation preserves distances and PH, but moving identities between regions can change labelled component membership. Negative nearest-opposite-minus-same values mean an opposite-label entity is closer than the nearest same-label entity. Smaller first-opposite merge distances indicate earlier target mixing, but are not themselves evidence of loss or causality. Null results are descriptive, not p-values. H0/H1 on a finite cloud is not a proof of the topology of the full model representation manifold.

### Temporal link

When consecutive updates are measured (20/21 and 46/47 by default), `lagged_example_predictions` joins changes known after the earlier update to per-entity A loss changes at the next update. It includes prior loss and prior loss change as baselines. This table does not fit a predictive model or claim held-out predictive performance. Multiple seeds and genuinely separate evaluation choices are needed before testing a predictive signature.

## Outputs

In the results root:

- `trajectory_report.txt` and `trajectory_summary.json/.csv`: compact actual-update results;
- `trajectory_updates.png`: observed/linear A changes and history/current contributions;
- `trajectory_examples.png`: task-sensitive motion and labelled separation against per-entity loss changes;
- `lagged_example_predictions.json/.csv`: temporal joins.

Within `trajectory_seed1/`:

- `trajectory_metadata.json`: exact source fork, steps and calibration scope;
- `A_anchor/` and `stateNNN/`: every entity's losses, coordinates, hidden arrays/Jacobians and labelled topology;
- `updateNNN/actual_update_geometry.json`: actual displacement, Adam channels and finite-difference audits;
- `updateNNN/example_geometry_changes.json/.csv`: per-entity geometry/loss changes;
- `updateNNN/task_ph_changes.json`: paired PH comparisons.

The notebook's optional EXPORT switch creates a ZIP of the entire results folder, printing its path. Share that archive to avoid ambiguous same-named files from different steps.

## Cost and scope

This is more expensive than a forward-only snapshot. Default OLMo probes process 128 entity/prompt combinations per state with two backward sensitivity calculations each. Only selected updates receive these probes. Each measured step also computes the full A-test gradient and four finite-difference evaluations.

Host memory temporarily holds roughly two full FP32 parameter-sized collections (about 9.4 GB for the 1.177-billion-parameter model), plus working copies and loaded artifacts. GPU memory includes the model, Adam states, gradients and activations. Real OLMo GPU runtime/memory have not been measured here. Reuse the existing working environment rather than reinstalling torch automatically.

The experiment tests an artificial association task in original OLMo. It does not establish a general mechanism of LLM forgetting, does not perform an inverse-rotation rescue, and does not treat a spectral or topological change as causal evidence. The purpose is to identify task-relevant motion that can be followed by a specific intervention.


Checkpoint input validation follows the enabled experiment. Trajectory-only runs ignore optional extra HF checkpoints and fingerprint the actual earliest trajectory fork, not the unused checkpoint-study event list. Active remote HF checkpoint comparisons still require a pinned revision; nonexistent absolute local paths produce an explicit missing-directory error.
