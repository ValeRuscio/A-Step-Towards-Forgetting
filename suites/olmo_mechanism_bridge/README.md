# OLMo mechanism bridge

This package connects the observed selection/content interaction to attention computation, task-sensitive directions, alignment interventions, optimizer continuations, and topology. It uses your **original OLMo association campaign**, including its optimizer forks and deterministic minibatches. It does not train a new language model from scratch.

## Start

1. Extract the entire ZIP into a new directory on the GH200. Keep all Python files next to the notebook.
2. Open `Run_mechanism_bridge.ipynb` in your existing working OLMo environment.
3. Run **Configure**, then **Run / resume**. The source is normally found at `/home/ubuntu/1/runs/olmo_association_v1`. If needed, edit only `SOURCE` in Configure.
4. Rerun **Refresh status** or **Show results** when you want an update. No output streams continuously into the notebook.

Use the original directory containing `config.json`, `seed*/data.json`, and `seed*/fork*/fork.pt`. A measurement ZIP or overnight result directory cannot replace the original checkpoints. No new remote checkpoint is requested. The original model/tokenizer revision is reused.

The default session allows 11.5 hours, with five minutes reserved for reporting. The queue is deliberately larger than one night. When it reaches `budget_paused`, run **Run / resume** again to grant another session. There is no claim that every job will fit in 12 hours.

## Stop, resume, restart

The notebook's last cell has an `ACTION` setting:

- `status`: inspect progress (default, safe when using Run All).
- `stop`: stop this package's verified background worker. Finished units remain.
- `resume`: continue in the same output directory.
- `restart`: stop and start a fresh sibling directory, preserving earlier results.
- `export`: refresh the report and ZIP.

Repeated launch is harmless. Changed scientific settings or package code automatically select a new output folder. Runtime hours/disk reserve can change without invalidating completed units. The notebook records the actual output directory in `bridge_session.json`, including after restart.

A forcibly interrupted prompt may be recomputed. Continuations save model and optimizer state after every completed, traced update. Interrupted steps replay from the preceding committed state. Source checkpoints are never overwritten. A killed process may leave a temporary checkpoint; these files are not treated as completed states.

## What runs

### 1. Factorial parameter interventions

At each saved update fraction, evaluate four corners: neither group, selection only, content only, and both. Selection comprises Q/K weights and MLP gate weights. Content comprises V/O, MLP up/down, and embedding/readout/other weights. Normalization parameters, if present, remain at the same fraction in every corner.

For each corner, save every entity's target KL, signed two-token margin, target margin, relation loss, and probability-support loss on A/B valid/test. Capture all layers by default:

- RoPE-audited attention probabilities P and values V, including all query positions;
- attention output before the current O projection;
- normalized Q/K inputs and projected Q/K;
- MLP gate preactivations and layer residual output;
- exact local loss and margin derivatives with respect to these hooked quantities;
- final normalized hidden representation.

The parameter factorial is distinct from crossing activation operands. Changes in Q/K weights are not identical to changes in P, because upstream representations also change. Likewise, V activations can move without a large change in V weights.

### 2. Fixed-map intervention within every corner

Interventions initially target recorded layer indices 0, 1 and 9 (zero-based). All other layers are still observed. Set `patch_layers=[]` to intervene at every layer, at considerable additional cost.

Fit a **single proper orthogonal map** from the 11 corner toward the pre-update activation, using only A/B valid prompts from a deterministic calibration subset of entity identities. Use at most four positions per calibration prompt and a joint-span rank cap of 64. No target losses or test prompts fit the map. The transform is identity outside its fitted span; it has no translation. The same map is applied to every corner. It rotates the concatenated attention-output coordinates, potentially mixing heads; it is not a Q/K RoPE-plane rotation or a temporal phase measurement.

Compare:

- sham reconstruction;
- earlier routing, earlier values, and both;
- rotation-only correction;
- calibration-fitted nonnegative scalar correction;
- calibration-fitted mean-shift correction;
- random additive displacements matching the rotation correction's activation displacement separately for each prompt/corner;
- random additive displacements matching earlier-both replacement;
- random orthogonal controls: conjugate the fitted rotation in the same calibration span. These preserve vector norms and the rotation's eigenangles/operator distance from identity, **not** each prompt's displacement magnitude. The additive control supplies that complementary match.

Maps are fitted separately for each seed/event/fraction/layer, then held fixed across corners. Failed fits or reconstruction audits fail the job explicitly. Finite-difference checks of margin directional derivatives are recorded on calibration prompts. A failed derivative check is recorded, not silently counted as confirmation.

All query positions are patched before the current O projection, leaving current downstream weights fixed. Earlier operands are counterfactual interventions, not realistic training updates. Corrections can be out of distribution. A loss rescue and reduction of the factorial interaction provide evidence about the tested intervention; they do not by themselves prove unique causal mediation or identify a universal cause.

### 3. Task-sensitive accounting and topology

Decompose the activation change exactly as:

`Δ(PV) = ΔP V0 + P0 ΔV + ΔP ΔV`.

Save each term's norm and projection onto pre-update/current margin derivatives, and onto the pre-update loss derivative. Record reconstruction errors. For every finite patch, save its exact loss/margin change, local linear prediction, and remainder. The factorial **loss** interaction can arise downstream even when the direct activation cross-term is small.

Save per-entity attention entropy, gate-sign changes (a preactivation sign diagnostic, not binary gating), and residual changes.

For fixed, held-out entity landmarks, compute exact H0/H1 flag persistence over F2 for up to 32 points. Use baseline distance scales, not independently normalized scales that erase changes in size. Record covariance spectra and effective rank alongside covariance-preserving point-mixing nulls. This is an empirical connection to spectral structure, **not an assumption that Marchenko–Pastur laws apply** to these trained activations. No asymptotic RMT p-values are claimed.

A second filtration uses a frozen, calibration-fitted, target-weighted margin-gradient semimetric at the final token of the attention output. It is not the full vocabulary Fisher, Hessian, NTK, or an all-position metric. The metric is unchanged across corners.

Also compute Euclidean PH at the patched site and final hidden representation after each intervention. A rigid rotation preserves patched-site Euclidean PH. It may nevertheless change downstream representations and loss. Thus a rotation rescue does not require a topological change at the rotated site.

Persistent homology is descriptive here; there is no validated topology-based predictor. Point-mixing nulls disrupt entity associations, and their purpose is to test structure beyond a covariance spectrum. A few null repeats are not a statistical discovery threshold.

### 4. Optimizer dynamics

Replay original, reset-first-moment, and quarter-learning-rate branches using identical scheduled B minibatches for 16 updates. The first-moment reset preserves the second moment and Adam clock. Save every step's A/B losses, peaks/step-averages/final values, native history/current numerator accounting, and exact A-gradient/actual-update alignment. Instrument the selected patch layers along continuations (all layers can be enabled with `patch_layers=[]`).

Save geometry and task-direction records at every step by default. Branch geometric references remain fixed at the fork; optimizer linear accounting is local to each update. Those are different reference points.

A separate experiment compares the original and reset-m first-step directions at equal total parameter displacement from identical weights. Fractions 0.25 and 1.0 are evaluated. These are single displacements, not norm-matched training branches, and do not ensure equal progress on B.

`trajectory_features.csv` joins entity losses to computational features so preceding-step features can later be evaluated as predictors on unseen seeds/events. The supplied report does not fit or claim a validated predictor and does not infer temporal phase from an oscillating loss curve.

## Calibration and evaluation

Half of entity identities, chosen by a fixed seed, calibrate the corrections using valid prompts only. Held-out-entity results exclude those identities from **every** split. Here held out means excluded from repair calibration, not excluded from the original model training. All-entity results are provided separately for continuity with earlier reports. A-test still has a different prompt template; calibration does not see it. The same entity's multiple prompts are not independent replicates.

These seed/event choices were informed by previously inspected experiments. New held-out-entity interventions are useful controls, but these events are not a pristine confirmatory sample. Replication at other events and new seeds remains necessary. The code does not search test losses to select a best repair.

## Outputs

`SETTINGS['output']` is the authoritative output folder. The notebook prints it. **Show results** also creates `bridge_results_share.zip` beside the notebook and displays a working download link.

- `REPORT.txt`: numbers and interpretation limits.
- `interaction_summary.json/.csv`: four corner means and interaction for every intervention; all/held-out populations separated.
- `factorial_*/entity_interactions.json`: paired per-entity interaction contrasts.
- `branch_summary.json/.csv`: final, maximum and mean-across-saved-steps losses.
- `trajectory_features.csv`: joined continuation computations and losses.
- `interactions.png`, `repair_interactions.png`, `continuations.png`: plots.
- `factorial_*/frame_layer*.npz`: fitted map, scale, mean shift and calibration fit error.
- `factorial_*/00|10|01|11/capture`: raw activations/derivatives and per-prompt losses.
- Corresponding intervention folders: exact counterfactual outcomes, finite-patch projections/remainders, patched-site/final-hidden PH.
- `directions_*/directions.json`: equal-displacement optimizer comparisons.
- `branch_*/step*/update.json`: native Adam channel accounting.
- `source_fingerprints.json`, settings, hardware and status records.
- `results_share.zip`: generated automatically at pause/completion. By default includes JSON/CSV/plots, **not** raw NPZ or model/optimizer checkpoints. Set `export_arrays=True` to include arrays; checkpoints are always excluded.

Long jobs save per-prompt units before the top-level job finishes. Partial jobs are visible in status; their absence from aggregate interaction plots is not a negative result. The plots display completed comparisons only. Reports can be refreshed while a job runs.

## Resources and validation

Use the existing working FP32 OLMo/CUDA environment; do not blindly reinstall PyTorch on a GH200. `requirements.txt` records the compatible package families. Python 3.9+ and the original OLMo architecture are supported; OLMo2/3 are not silently substituted.

A 1B model's FP32 model+Adam checkpoint is roughly 12 GB. Each continuation retains one checkpoint; an atomic commit temporarily needs another copy. Three seeds × three branches may therefore retain about 108 GB in optimizer checkpoints alone, plus captures, temporary writes and the source campaign. Plan for at least **150–200 GB additional disk**, with more if you enable every layer intervention or include raw arrays in exports. Runtime and storage depend on prompt lengths. The disk reserve pauses the run instead of discarding data.

There are no background model experiments on the author's machine. The package is tested with a tiny randomly initialized original-OLMo fixture, including real forward/backward passes, attention reconstruction, factorial patches, rotations, PH, optimizer continuations and completed-job resume. This verifies implementation paths, not scientific results or full-scale GH200 performance.

Run optional local validation with:

```python
# From a terminal in this package directory:
# python test_bridge.py --smoke-output /path/to/new/tiny_validation
```

Keep `mechanism_bridge.py`, `bridge_core.py`, and `bridge_replay.py` together. The latter two vendor the original replay/audit implementation under distinct names so they do not collide with older notebook modules. Your previous experiment code and results remain usable.
