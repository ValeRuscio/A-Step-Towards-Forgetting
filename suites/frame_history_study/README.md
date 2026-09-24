# Reference-frame change × optimizer history

This standalone package tests whether a fixed, held-out activation-frame correction changes both task performance and the finite interaction between the history and current-gradient contributions to an actual Adam update. It uses the ORIGINAL OLMo association checkpoints. No new training dataset, model download revision, optimizer reset or continuation checkpoint is created.

## Run

1. Upload/unzip this folder on the GH200. Keep all Python files together. Open `Run_frame_history.ipynb` in your existing working OLMo environment.
2. Set SOURCE to the original association directory, normally `/home/ubuntu/1/runs/olmo_association_v1`. A measurements-only share ZIP cannot replace it.
3. Launch. Refresh status or Show results only when you want output. Stop is cooperative; after alive=False, Launch resumes saved units. Scientific settings changes create a separate output directory rather than mixing results. The notebook remembers the last output.
4. Use Export after stopping/completion. The share ZIP contains all unit records, fitted maps, settings, source configuration and copied code.

Run this after stopping/completing the other GPU suite: the two workers are separate and do not coordinate GPU memory. This release is for original OLMo, not the Pythia/Qwen adapters. It does NOT modify or resume the multimodel experiment.

## Primary question and design

At one naturally occurring B update, reconstruct the exact pre-event weights and Adam state. Capture the native signed parameter-displacement channels h (history) and c (current minibatch), with the SAME updated Adam denominator, bias correction and learning rate. The observed displacement is delta=h+c+r; save and audit the floating-point residual r.

At each fraction alpha, probe the four corners:

    theta(u,v) = theta_before + alpha * (u*h + v*c + u*v*r), u,v in {0,1}.

The full corner reproduces the actual update segment up to floating-point evaluation differences. The small residual enters the interaction explicitly; it is not silently discarded. These are controlled parameter probes with a frozen denominator, not four independently recomputed Adam steps.

For each task and activation condition, compute:

    I = L11 - L10 - L01 + L00.

Also record L11 itself, all four corner losses, per-entity outcomes, and the patched-minus-unpatched changes in L11, L00 and I. A reduced interaction alone can result from damaging a different corner. Require an actual held-out A endpoint improvement, examine all corners, and report the B cost.

## Frame correction

Site: final-token residual output of each selected transformer block, including that block's residual additions. Other token positions are untouched. A hook changes this output only for the diagnostic forward pass; no weights or optimizer states are updated.

Fit one **uncentered proper orthogonal Procrustes map** from natural POST-event activations to PRE-event activations, using A_valid and B_valid prompts from half the entity identities. The map acts in their joint row span and is identity on its complement. This fixes the otherwise underdetermined high-dimensional fit in an explicit way. It preserves norms about the origin; it is not a centered affine Procrustes transformation. Calibration rank and complete map are in maps/*.npz.

The SAME map is held fixed across all four corners and all alpha values for that event/layer. Evaluation uses A_test/B_test prompts from the disjoint half of entity identities. There is no fitting to test loss or choosing the best map from test outcomes. Layer/event selection remains exploratory: the defaults revisit previously inspected events. Do not turn the best test layer into a confirmatory claim without fresh validation.

Conditions:
- Unpatched and sham.
- Fitted rotation and its inverse.
- Calibration mean shift only; calibration global norm scale only.
- Four random orthogonal controls: conjugates of the fitted low-rank rotation in random subspaces, preserving the full rotation's eigenangle spectrum (including identity complement). Activation displacement norms need not match.
- Four additive controls: deterministic random directions per entity, norm-matched to the fitted rotation's displacement for that prompt at THAT corner. These are perturbation-magnitude controls, not common rotations or spectrum-matched controls.

The map uses both tasks' calibration activations without target-loss optimization. Because activations alone may not identify a functional frame, held-out loss effects and the controls are the primary test.

## Saved measurements

Each completed split/condition/corner is an atomic JSON unit; a pause does not invalidate it. No raw gradient or prompt-activation histories are saved. Fitted compact maps are retained.

- Full-population pre-event four-vector Gram matrices/norms/cosines, globally and per parameter layer; native channel reconstruction residual and pre/post weight hashes.
- Per held-out entity: KL, full-vocabulary majority-answer probability, probability mass on the two answers, answer-restricted accuracy, full-vocabulary majority-label top-1 accuracy, signed answer margin and actual patch magnitude. With .9/.1 soft targets, majority-label accuracy is complementary to KL, not the training objective.
- Held-out alignment error relative to the same entity's pre-event activation.
- Centered activation covariance eigenvalues, mean activation norms and recent pairwise-distance drift.
- Exact finite-cloud H0/H1 Vietoris–Rips persistence over F2 (including all triangles), for up to 16 fixed held-out entities per task. The same reference distance scale is used across conditions/corners for one event/layer/task. Full intervals and Betti curves are retained; an empty H1 is not proof of no geometric change.
- The same metrics for a fixed three-category output representation: square roots of probabilities for answer token 0, answer token 1, and all other tokens aggregated. Euclidean distance is proportional to coarse Hellinger distance. It is not full-vocabulary Fisher geometry, the loss Hessian, or an NTK.

A rigid correction at the measured activation site should preserve its Euclidean persistence apart from numerical errors, EVEN IF downstream loss and output geometry change. This is a useful invariance control, not a failure of the experiment. Output geometry is coarsened and does not uniquely identify the mechanism.

## Defaults and runtime

Events: seed1/update21 (persistent deterioration), seed1/update20 (adjacent comparison), seed1/update48 and seed3/update7 (recovering episodes relative to their immediate pre-event baselines). The comparison event is not assumed equivalent in every respect. Layers 0,1,9; fractions .25,.625,1; four random controls of each kind. Default 11.5-hour session, 10-minute reserve, 25 GiB free-disk reserve, 5 GiB output guard.

For 32 original entities these settings produce 4,032 split-condition units, each with 16 held-out prompt forwards, plus calibration and gradient work. No GH200 runtime is claimed. Probes use one prompt per forward to stabilize control identity and avoid batch-dependent ambiguity. Resume repeats checkpoint reconstruction/calibration but skips completed units. Storage is bounded and dominated by small JSON records and maps, not model checkpoints. A source replay plus full gradients can require substantial host RAM; GH200 is the intended machine.

To run a smaller FIRST pass, configure one event, one layer, and alpha=1 before launching a NEW output. Those choices answer a narrower question; do not silently change settings inside an existing run.

## What this can establish

A fitted map that improves held-out A outcomes, behaves better than the matched controls, and selectively attenuates the h/c interaction provides causal evidence about that diagnostic interface and parameter-plane experiment. It does not prove that a natural frame rotation generated the original optimizer history, that the map repairs the model permanently, or that the result generalizes across LLMs.

This package does not run continued training branches, estimate new curvature/HVPs, compute activation-sensitivity gradients, perform RMT hypothesis tests, or prove persistence under a repair. Those are distinct experiments. The existing natural trajectories and curvature audits remain complementary.

## Dependencies and validation

Use the existing torch/transformers/numpy/matplotlib environment; do not upgrade Transformers to run this package. Topology is bundled and requires no ripser installation. See VALIDATION.txt for the exact local tests and limits. Results are exploratory; four random controls do not support small p-values.
