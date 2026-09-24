# Forgetting mechanism experiments

Run `Run_forgetting_mechanisms.ipynb` alongside `forgetting_mechanisms.py`. All model, data, experiment, plotting and process-control functions are in that one Python file. No imports from the previous experiment directory are required.

## Notebook workflow

1. Open the notebook in your existing OLMo environment. Set `SOURCE` to the original `olmo_association_v1` campaign, and choose a **new**, separate `OUTPUT` directory.
2. Run **Launch** once. It returns immediately; work runs in a separate process with output saved in log files.
3. Rerun **Refresh status** whenever you want to check progress. There is no automatic polling or live output stream.
4. Rerun **Show results** to print the compact numerical summary and show plots for completed cases. Set `FULL_REPORT=True` there for every intervention's numbers. Full CSV/JSON and `report.txt` are saved regardless.
5. An optional, disabled stop cell requests termination. Launch again with identical settings to skip completed cases and restart unfinished cases. It does not resume halfway through a case.

A notebook kernel restart normally leaves the detached process running on Linux/macOS. Restarting the machine or terminating its notebook-server/container can terminate it. A heartbeat demonstrates process liveness, not mathematical progress. The status includes the current phase/direction/dose so you can distinguish a lengthy operation from completed work.

Default production jobs: seeds 1, 2, 3 at events 21 and 47. To reproduce the wider diagnostic schedule, set `events` to `[20,21,22,46,47,48]`. Default finite doses are 0, .25, .5, .75, 1. Test prompts are previously inspected, so these are exploratory follow-ups, not confirmatory replications. Different seeds/events may have different mechanisms.

## Input and environment

The production source must contain `config.json`, and each selected `seedN/data.json`, `seedN/anchor.pt`, and a preceding `seedN/forkNNN/fork.pt`. **The measurement ZIP alone contains predictions, not model weights or optimizer state, and cannot run new interventions.** It can run the separate saved-data analysis.

The checkpoint core was copied from your local `forgetting_measurement/association` implementation. Data hashes, checkpoint step labels, model revision and full-step replay are checked. The fork convention is the original convention: a fork at event t is the state immediately before the minibatch indexed t. PyTorch checkpoints use `weights_only=True` loading. The runner does not modify the source campaign.

Use your existing working PyTorch/CUDA installation. The code uses the original OLMo architecture and requires a pinned 40-character revision in the production source config. Packages are listed in `requirements.txt`; do not replace a working CUDA PyTorch installation just to run this notebook. The runner supports Linux/macOS process management; production is intended for the original Linux GPU machine.

Directions are stored on disk and loaded sequentially. Preflight estimates free host RAM as about 10 times model parameter bytes plus 2 GiB and free disk as about 18 times parameter bytes plus 1 GiB. For a billion-parameter FP32 model this is substantial. These are planning estimates, not guarantees; GPU memory depends on the original model and batches. Temporary directions and the continuation checkpoint are removed after each successful case by default. Failed cases retain their diagnostic files. Model construction happens before the detailed preflight and requires the original inference/training hardware.

The full campaign has not been timed or run on OLMo here. A case involves many forward evaluations, full-panel gradients and disk reads. Begin with one seed/event if you want a timing estimate, then use a **new output directory** when expanding the design.

## What is tested

### Actual-step and layer-0 dose

The actual path is `theta_pre + alpha * actual_delta`. The layer-0 path is `theta_pre + delta_other + alpha * delta_layer0`. Thus alpha=0 in the second path removes layer 0 only; alpha=1 reconstructs the full update. Both references (pre-event and learned A anchor) are retained in numerical tables. A finer actual-step curve is also saved.

### Optimizer contributions

The actual clipped minibatch gradient is used. The first-moment numerator is split into historical `beta1*m_previous` and current `(1-beta1)*g` contributions. Both use the **same updated second moment, timestep and bias corrections** as ordinary Adam. Their sum is audited against the applied displacement.

Native-contribution effects are saved separately in `native_optimizer_channels.json`. The main comparison also normalizes each nonzero direction to the full actual update norm, then applies the dose sweep. SGD-like and full-B-training-gradient directions and an actual-magnitude sign-scrambled control are included. These are common-state direction interventions, not native-optimizer benchmarks. This does not recompute historical gradients or prove gradient staleness.

### Loud and quiet directions

Let V contain the normalized candidate parameter directions and let J map their perturbations to answer log-odds on a small panel of **training prompts**. The runner measures `G = V.T V` and response derivatives, whitens G, and computes the singular values of the resulting restricted Jacobian. This avoids calling a direction loud merely because the chosen basis vector was longer or duplicated. Unique model parameters are counted once, including tied embeddings/output weights.

Each finite-difference column is checked at two spacings, against parameter injection error, and against two independently differentiated scalar output probes. Numerically flat or failed columns are excluded. These scalar checks supplement, but do not constitute exhaustive componentwise autodiff verification. Every derived loud/quiet/target-fit direction is then independently checked against its predicted response at two finite spacings, since cancellation can amplify errors. Failed derived directions are excluded from the finite comparison. If too few columns resolve, the loud/quiet stage is explicitly unresolved; the other experiments still run. No thresholds are silently relaxed.

Loud modes account for the first 90% of measured response energy, reserving a lower-gain mode when possible. The runner tests normalized projections of the actual update into loud and quiet modes, a quiet projection of the B-training descent direction, and a training-label oracle joint-target fit. Vanishing projections are omitted. A rank-one span may have no quiet comparator.

`spectrum.json` also records whether each response mode changes margins uniformly or differentiates registries, how the desired joint training-margin correction projects into the loud/quiet response spans, and the predicted minimum norm needed for its representable part. The joint-target comparator is norm-matched for finite tests; its possibly much larger unconstrained linearized norm is reported separately. The oracle fits conditional margins, not full-vocabulary KL.

**This is a spectrum in a small, explicitly listed parameter span. It is not the full model Jacobian, Fisher matrix, Hessian, or global memory capacity.** Failure to find a useful quiet direction cannot establish that none exists. Large representation Gram eigenvalues from your earlier archive are also a different quantity.

### Validation matching and continuation

Candidate selection uses A/B **validation** full KL only. Candidates must match the full nominal update's B-validation improvement within 10% or 0.0001 KL, whichever is larger; among matches, the lowest A-validation damage is selected. A nonpositive nominal validation gain or no match is reported explicitly. There is no fallback that selects on test results. Test numbers for the predeclared grid are recorded for exploratory inspection.

The selected displacement and nominal displacement are continued for 16 B updates by default. Both inherit the exact **same nominal post-event Adam state**. This isolates the consequences of changing weights, and is not continuation with a permanently reset or surgically changed optimizer. The selection may fail to generalize; that is a valid result. Native optimizer-state intervention/washout experiments would require an additional protocol.

All interventions report the actual training minibatch, ordinary A/B test KL, label-balanced A/B test KL where both labels exist, and pre-event/anchor references. Missing label classes produce an explicit null balanced value. No confidence interval treats the 18 correlated events as 18 independent seeds.

### Token traces

`token_traces.json` contains layer-0 q/k/v/o and MLP output RMS at **every prompt position**, for fixed training calibration prompts at pre-event, quarter-step and full-step states. These are descriptive localization traces; they do not themselves measure attention probabilities or establish causal mediation through a token or head.

## Output files

Each `seedN/eventNNN/` contains:

- `overview.png`: dose response, test A/B trade-off, restricted spectrum, and entitywise log-odds shifts.
- `interventions.csv`, `interventions.json`, `intervention_predictions.json`: all finite comparisons and per-entity predictions.
- `native_optimizer_channels.json`, `fine_dose.json`, `continuation.json`, `selection.json`.
- `outputs.json`: anchor, pre-event, post-event predictions and optimizer-sum audit.
- `calibration.json`, `calibration_response.npz`, `spectrum.json`, `derivative_audits.json`, `replay_audit.json`.
- `token_traces.json`, `protocol.json`, `status.json`, `worker.log`, `done.json`.

The campaign directory contains `all_interventions.csv`, `report.txt`, source/code/version fingerprints, settings, status and launcher logs. Changing inputs, code, settings or versions requires a new output directory.

## Saved-data analysis

The notebook has an optional cell to reanalyse `share_forgetting_measurement.zip`. It prints no long report during analysis; use its results cell to see the numbers and plot. Included `example_results` were actually computed from your supplied measurement archive. They are **not** results of the new production interventions.

Command line equivalents:

```text
python forgetting_mechanisms.py analyze --measurement /path/share_forgetting_measurement.zip --output /path/reanalysis
python forgetting_mechanisms.py self-test
python forgetting_mechanisms.py smoke --output /path/new_tiny_test
python forgetting_mechanisms.py run --settings /path/settings.json
python forgetting_mechanisms.py report --output /path/experiment_output
```

## How to interpret the loud-direction hypothesis

A concentration of sensitivity could limit what is accessible under a fixed step-size, training-time or noise budget. In a local linear model, a desired output correction along singular mode i requires a parameter displacement proportional to `correction_i / singular_value_i`. Quiet directions can therefore be expensive without being absent.

Evidence for an accessibility bottleneck would include a concentrated audited spectrum, old/new target corrections competing in the accessible modes, and difficulty obtaining matched B gain with less A harm even after enlarging the tested span. Evidence against a necessary trade-off at an event would include a finite held-out comparator that improves both tasks or preserves A at comparable B gain. Neither outcome establishes a universal information-storage limit.

The quarter-step result in your existing seed-1/event-21 data already weakens the claim that its observed full-step damage was necessary for the measured B gain. Learning-speed differences across kernel eigendirections are established in the NTK regime; extending that result to a hard memory bound in finite OLMo requires additional evidence. Reference: https://arxiv.org/abs/1912.01198
