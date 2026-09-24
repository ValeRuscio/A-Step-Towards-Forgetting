# OLMo forgetting: geometry along actual Adam updates

Full code, using the original quiet notebook workflow. Keep `structure_study.py`, `forgetting_mechanisms.py`, and `Run_structure_study.ipynb` together. No patch cells or separate runner are needed.

## Run it

1. Extract this ZIP into a new folder on your GPU machine, using your existing working OLMo environment.
2. Open `Run_structure_study.ipynb` and run the setup and configuration cells. It looks for your original `olmo_association_v1` campaign, including `/home/ubuntu/1/runs/olmo_association_v1`. If it cannot find it, set `SOURCE` to the campaign containing `config.json`, `seed1/anchor.pt`, `seed1/data.json`, and saved `forkNNN/fork.pt` files. A previous measurements ZIP is not the source.
3. Run **Run / restart**. The process runs in the background; the cell returns immediately.
4. Rerun **Refresh status** for progress, or **Show results** for the numbers and plots produced so far. There is no continuously printing cell.

To stop, set `STOP=True` in the Stop cell and execute it. Set it back to `False` afterward. To run again, execute Run / restart: it stops the tracked run, creates a fresh output folder, and preserves previous results. `study.launch(SETTINGS)` can instead resume a compatible stopped run. A kernel restart loses the in-memory output selection; retain the printed results path if a background run is still active.

The notebook's export cell can ZIP the entire results folder for sharing. The actual results folder is always `SETTINGS["output"]`, printed when the run starts.

## What runs by default

Seed 1, original B updates **20 and 21**. Set `SETTINGS["trajectory"]["steps"]` to include 46/47 or other saved-schedule updates. The original Adam state, clipped training gradients, minibatches and model backend are replayed from the preceding saved fork. Original source checkpoints are read only.

For each selected update, let `delta = theta_after - theta_before`. The experiment evaluates

`theta(alpha) = theta_before + alpha * delta`, for `0 <= alpha <= 1`.

These are fractions of **one fixed, actual update**, not additional optimizer steps. At each sampled fraction it measures:

- A/B loss for every test entity;
- the full A-test gradient dotted with `delta`, giving the local A-loss slope along this path;
- independent central finite-difference slope checks at two spacings, with individual entity estimates and spacing disagreement;
- task-sensitive hidden geometry, answer/decoder angles and norms, and the exact final answer-margin decomposition;
- H0/H1 persistence and target-labelled neighbourhood/component measurements in Euclidean and frozen A-anchor Fisher distances;
- reconstructed attention distributions, Q/K alignment and relative RoPE-plane phase, subject to an attention-value-output audit.

There are nine initial fractions, plus at most six extra fractions bisecting intervals with opposite-sign A slopes. Refinement stops when observed brackets are narrow enough or its budget is exhausted. It can miss reversals entirely between sampled points. A sampled minimum is not a guaranteed global minimum. The two finite-difference checks intentionally probe slightly outside `[0,1]` at the endpoints.

The key question at step 21 is **where an initially favourable A direction becomes unfavourable, and what changes geometrically near that location**. Step 20 provides a contrasting update. The code does not assume either pattern will reproduce: it records the measured curves and audits.

## Geometry and topology

All source entities are retained: ordinarily 32 for each of A_valid, A_test, B_valid and B_test. Default sites are the first, middle and last block outputs plus final hidden state, at the answer-producing final token. Explicit layer indices can be supplied with `SETTINGS["trajectory"]["layers"]`.

The frozen task metric is fitted once to all A-anchor validation entities and retains the same distance scale at every path point. It uses the Fisher geometry of the three outcomes answer0/answer1/other. It is a semimetric, not the full-vocabulary Fisher or the target-loss Hessian. At intermediate blocks, sensitivities hold other token positions fixed; they are not complete causal attributions of network changes.

Persistence is computed directly over all entities for H0 and H1, without a ripser dependency. Labels annotate component mixing and neighbour separation. Label-permutation controls preserve class counts. Missing opposite/same-class neighbours are recorded as undefined, not zero. These finite-cloud measurements do not establish the topology of a full representation manifold.

The phase diagnostic uses a fixed first-key position and the phase of a weighted complex Q/K cross-product across RoPE planes. It is a spatial attention diagnostic, not temporal optimizer phase. Small aggregate resultants make that phase undefined. The reconstruction checks `P @ V` against the model's actual attention output before its output projection. This is a useful consistency audit, not a proof that each recovered probability is exact. Failed audits suppress attention-derived features and remain visible in `path_attention.json`.

Harmed/not-harmed groups in the geometry figure are defined retrospectively by endpoint A loss. They are descriptive. `next_interval_features.csv` pairs current features with the following loss increment; it contains no fitted model and excludes the retrospective endpoint-loss label. It must not be treated as a validated predictor or as independent observations across sites/entities.

## Where the results are

The results root contains `path_report.txt`, `path_summary.json`, and the existing trajectory summaries and plots. Each `trajectory_seed1/update020/path/` (and update021) contains:

| File | Contents |
|---|---|
| `loss_and_slope.png` | A/B loss changes and exact A slope with its numerical check |
| `geometry_along_path.png` | Old loss-sensitive motion, sensitivity orientation and labelled separation |
| `loss_curve.json/.csv` | Sampled fractions, A/B loss, A slope and audits; JSON includes per-entity derivatives |
| `path_summary.json` | Endpoint changes, slope-reversal brackets, sampled minimum and audit counts |
| `all_entity_features.json/.csv` | Geometry and retrospective endpoint labels across fractions |
| `next_interval_features.csv` | Current measurements joined to subsequent loss increments |
| `pointNNN/` | Raw hidden states/Jacobians, identities, topology, PH changes, attention audits and features |

Each point's `path_position.json` records its actual fraction. The earlier before/after measurements and native Adam history/current decomposition remain in the parent update folder and the trajectory state folders.

The optional original checkpoint PH/RMT/NTK study remains available through `run_checkpoint_study=True`. It is disabled by default to focus computation on the update path. Unused extra HF checkpoint specifications cannot block trajectory-only runs. Actual remote checkpoint comparisons still require pinned revisions.

## Cost, checks and interpretation

This is substantially more expensive than endpoint comparison. At up to 15 fractions per update it performs an exact A-panel gradient, five A/B loss evaluations, and geometry probes for 128 prompt/entity combinations with two backward sensitivity calculations each. Attention adds forward passes. Start with seed 1 and updates 20/21; runtime on your full OLMo GPU has not been measured here.

CPU memory temporarily holds roughly two full FP32 parameter collections (about 9.4 GB for 1.177 billion parameters), plus working copies and artifacts. GPU memory includes the model, Adam states, gradients and activations. Keep your existing compatible environment; do not automatically replace its CUDA/PyTorch installation.

Absolute weight assignments avoid cumulative interpolation error. A `finally` block restores the actual post-update weights; diagnostic gradients are cleared and optimizer moments are untouched. FP32 interpolation and differentiation remain numerical computations, so inspect the finite-difference audit rather than assuming exact arithmetic. An average slope audit does not establish every entity derivative's accuracy.

Local tiny-OLMo validation is documented in `VALIDATION.md`. The real saved OLMo checkpoint campaign was not executed here. These results can locate a geometric transition associated with forgetting; identifying a cause still requires a subsequent controlled intervention and replication. No inverse-rotation rescue is claimed.

See `TRAJECTORY_METHODS.md` for the existing before/after definitions and `CHECKPOINT_METHODS.md` for the optional checkpoint analysis.
