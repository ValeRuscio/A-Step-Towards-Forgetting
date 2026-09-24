# Natural forgetting geometry — overnight runner

This package follows the original OLMo association A→B training with no activation patches, altered learning rates, or experimental optimizer resets. It uses your saved A-trained anchor and the original deterministic B minibatches. If the original experiment configured an optimizer reset at the A→B transition, that original policy is reproduced. Existing research files are read-only.

## Start tonight

1. Unzip all files into a new directory on the GH200 machine, retaining the existing working Python/CUDA environment.
2. Open `Run_Natural_Geometry.ipynb` with that environment's kernel.
3. Run Setup, Settings, and Start. The default source is `/home/ubuntu/1/runs/olmo_association_v1`.
4. The worker detaches from the notebook. Refresh the Status cell when desired; it does not stream output. You can disconnect your browser while the machine stays on. Do not shut down the instance.
5. The session lasts up to 11.5 hours, allowing 10 minutes for shutdown/reporting. A long indivisible GPU operation or checkpoint write can exceed that reserve. If the configured trajectories finish earlier, the run ends earlier. There is no artificial waiting or promise that every configured measurement fits in one session.

Defaults: all available original seeds (normally 1,2,3), B updates 1–128, seed rotation every 8 updates, checkpoint every 8, geometry every 4, mixed curvature every 16. You can shorten `end_step` or change these settings before starting. Each seed starts at its A-trained anchor; this is a fresh unmodified replay of the existing task, not new pretraining or a new independent task sample.

`launch(SETTINGS)` starts or resumes. If already running it simply reports that. Scientific settings changes automatically create a new sibling output directory, preserving the old one; the SETTINGS dictionary is updated to point there. Budget/disk limits can be changed for a resumed session. `stop(output)` requests a cooperative stop; refresh until `alive` is false. `restart(SETTINGS)` starts a fresh sibling run after the old worker has stopped. It never deletes research data.

The default mixed-curvature calculation is expensive. It uses two fixed A-monitor identities and can use substantial GPU memory. The tiny test covers the algorithm, not full-size GH200 memory/time. If a full-model run fails with GPU out-of-memory at that phase, use `curvature_every=0` in a new run; the rest of the gradient/geometry measurements remain available. The error is preserved in `error.txt` and `worker.log`. Do not change numerical precision or kernels silently to make a comparable replay faster.

## What is measured

- Every update: full-vocabulary conditional KL, answer-pair support loss, target probability and answer logit margin for all entities in A/B valid/test prompts.
- Every update: the full parameter gradient of A loss averaged over fixed monitor identities, plus clipped B minibatch gradients from the unchanged training procedure.
- Every update: native Adam history/current numerator channels under the same updated denominator; their norms, inner products, cosine, first-order A projections, and reconstruction of the actual floating-point displacement. The source optimizer is Adam with zero weight decay; unsupported variants are refused, not silently reinterpreted.
- Every unique weight matrix: norm and cosine before/after the update and gradient-channel statistics. Tied parameters are counted once. QK, gate, OV, MLP content and other parameter-group aggregates are recorded separately from the history/current decomposition. Parameter groups are not an activation-level routing/value decomposition.
- Every 4 updates: local final-token hidden states, loss gradients and margin gradients at every transformer layer's residual output and final normalized hidden state. These are passive derivatives, with other token positions held fixed. They are not a complete decomposition of changes caused by every token/layer simultaneously.
- Every 4 updates: mean shifts, centered variance spectra/effective ranks, principal subspace angles, local sensitivity/displacement projections, and monitor-fitted orthogonal alignment evaluated on other identities. Principal angles are subspace quantities; individual near-degenerate eigenvectors are not assigned a unique frame. The Procrustes fit allows reflections and is a descriptive fit, not a performed repair.
- Every 4 updates: H0/H1, identity-labelled minimum-spanning-tree edges and neighbour retention, in Euclidean distance and a frozen baseline monitor margin-gradient semimetric. Fixed landmarks and baseline scales are used throughout each seed. This is not full Fisher geometry or the loss Hessian. Covariance-preserving point-mixing nulls are saved, without p-values from only four draws.
- Every 4 updates: fixed coordinate sample spectra from every unique weight matrix and independent-column permutation nulls. This is an empirical random-matrix comparison, not a calibrated Marchenko–Pastur law test or full-matrix spectrum.
- Every 4 updates: post-update A-monitor gradient projected onto the actual displacement, allowing comparison with the initial directional derivative. The difference is a directional gradient change, not a direct Hessian eigenvalue.
- Every 16 updates: autograd mixed contraction hᵀH_A c on two monitor identities at the original pre-update weights. It does not change training. Its task population is narrower than the full A-monitor first-order gradient and is labelled in the record.

The fixed monitor identities and geometry evaluation identities are disjoint. They were not excluded from the original model training: these are diagnostic holdouts, not unseen learned associations. A/B valid and test prompts remain separate. Eight monitor identities and 16 geometry identities are the default, with full 32-entity loss evaluations retained.

## How this addresses the hypothesis

A reference-frame explanation requires a relationship between activations and the downstream computation, not merely a fitted rotation or a changing norm. The package records old/current task sensitivity dotted with activation displacement, sensitivity drift, and intrinsic topology separately. Changes can therefore be compared instead of assuming a rotational mechanism.

History/current channels are two terms of Adam's actual update. They are not the same decomposition as QK/gating versus value/content parameters. Both are labelled separately. Positive gradient projection means a locally harmful direction for A; finite loss can differ because of nonlinear terms.

`frame_prediction.json` compares next-update A-test loss prediction using update norm alone, then channels, then channels plus monitor frame summaries. Ridge penalty is fixed at 1; centering, scaling and coefficients use other seeds only. All models in that file use the same scheduled geometry points. `lagged_prediction.json` provides the denser channel-only comparison. These are exploratory held-out-seed checks with few seeds and a shared task format, not proof of causality or a deployable forgetting detector. An incomplete run may not have enough points and will return empty score lists honestly.

## Files and disk limits

- `REPORT.txt`, `trajectory.csv`, `trajectory.png`: readable summaries, numbers and plots.
- `measurements.sqlite`: one record per natural update, including per-entity, per-matrix and scheduled geometry/topology measurements. Reading example:
  ```python
  rows = study.records(SETTINGS['output'])
  rows[0]['channels']
  ```
- `seed*/baseline.npz` and `snapshot_*.npz`: compressed float32 last-token states and sensitivities, with explicit split/site keys. There is no precision reduction beyond the model's original FP32 activations.
- `seed*/resume.pt`: one rolling model+optimizer checkpoint per seed. Temporary atomic replacement requires space for one additional checkpoint.
- `code/`, `provenance.json`, `partition.json`, `hardware.json`: implementation and experimental provenance. The pinned source model revision is retained. Source checkpoints themselves are not copied into the output.
- `natural_geometry_share.zip`: compact export including the SQLite measurement snapshot and summaries, excluding checkpoint and NPZ arrays.

Storage defaults: 30 GiB free reserve and a 20 GiB measurement-file budget, plus rolling checkpoints. For a roughly 1B-parameter FP32 Adam model, expect approximately 12 GiB per rolling state and another 12 GiB for atomic replacement; actual size is checked from parameter bytes. Three seeds usually need under 100 GiB including reserves/measurements, not hundreds of GiB of per-prompt files. These are estimates, not a hard total quota. Measurement-budget checks occur at checkpoints and can overshoot by one chunk.

Full token activations, full historical model checkpoints and full Hessians are intentionally not retained. The package does not claim to save all possible information. Stopping in the middle of an optional geometry capture can leave that update's geometry marked incomplete; completed training/loss data are retained when a safe checkpoint can be saved. A crash or insufficient checkpoint space resumes from the previous durable checkpoint and recomputes later records deterministically.

## Verification

Run `python test_natural_geometry.py` in the same environment for an offline tiny-OLMo test. It needs no model download. Tests check exact equivalence with an uninstrumented training loop, channel reconstruction, mixed Hessian against finite differences, PH invariance, report/export, completed resume and rollback after a simulated crash. Full OLMo-1B/GH200 performance is not tested locally.

Dependencies are in `requirements.txt`. Reuse your existing working CUDA torch installation. Only install missing dependencies; do not replace CUDA torch just to launch the notebook. The delivered tests ran with torch 2.8.0 and transformers 4.57.6 on CPU.
