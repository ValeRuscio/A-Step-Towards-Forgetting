# Overnight replication and event-local history study

Extract the whole ZIP into a new folder and open `Run_Overnight_Replication.ipynb`.
All Python modules and the unchanged frozen predictor are included. No old run
is overwritten. The notebook is quiet: launch once, manually refresh status.

## What runs, in priority order

1. **Both new seeds, 12 and 13**, Pythia-410M, SST-2 sentiment -> AG News topics.
   Six trajectories per seed: natural, projection-triggered, loss-history-triggered,
   constant 0.25x LR, first-moment reset at 1x LR, and first-moment reset at 0.25x LR.
   The natural condition is the preserved-history 1x LR control. No predictor or
   alarm cutoff is fitted on these seeds. This phase finishes for both seeds
   before the next phase starts.
2. **Five random schedules per intervention budget per seed.** Each schedule has
   the same intervention count as its policy and uses the same LR multiplier.
   When both policies trigger the same number of actions, they share the same
   five controls; identical simulations are not run twice. With unequal counts,
   each budget gets its own five schedules. The budgets use no test outcomes.
3. **Event-local history branches** at the first two calibration-validation
   increases above 0.05 nats, plus fixed update 20. The fixed step is a time-selected
   comparison, NOT guaranteed to be nonforgetting. The selected events can improve
   test loss; those cases are retained and must not be relabelled as test forgetting.
4. **Re-audit seed 11/update 5/B-test curvature**, if the original complete run
   directory with its anchor and tokenized data is available. This reads the
   historical run without writing to it. A share ZIP alone is insufficient.
5. **Deferred frame/path replication** using the original fixed layers 0 and 1,
   calibration/evaluation split, selection rules, rotations and controls. This
   expensive phase comes last. `deferred_frames=False` omits it if chosen before
   first launch; leave it enabled to continue the queue over multiple nights.

If both policies reach the usual 16-action budget, the primary queue has 12 core
trajectories + 10 random trajectories. With differing budgets it can have up to
20 random trajectories. At most 24 event branches follow (two seeds x three
selected sites x four conditions), each 8 updates long or shorter near the end.
There is no promise of finishing all stages overnight. The default 11.5-hour
session stops cooperatively and resumes when Launch/resume is run again.
A GPU operation must complete before a stop/budget check can take effect.

## Event-local conditions

Every event is reconstructed from its original A anchor with the identical B
minibatches. Before branching, replay must reproduce saved per-example losses
on all four evaluation splits within the declared numerical tolerance.

The four branches start at the SAME pre-event weights:
- `preserve`: original first moment, second moment and Adam clock.
- `reset_m`: first moment zeroed; second moment and clock unchanged.
- `reset_m_global`: same reset, with its first ACTUAL displacement rescaled to
  match the preserved-history displacement's global Euclidean norm.
- `reset_m_layer`: same reset, matching each transformer block's displacement
  norm separately. Embedding/readout/other parameters form one additional group.

The first update uses the identical B minibatch. Matching is audited after
FP32 parameter assignment, with relative tolerance 1e-3 and absolute tolerance
1e-10. A zero candidate that cannot attain a nonzero target is refused. The
second moment and clock are checked before the first update. The shared anchor's
optimizer tensors are deep-copied, preventing branch-to-branch state contamination.

Norm matching changes the first displacement only; optimizer moments advance
normally for the selected branch. Later steps use the original nominal LR.
Consequently the immediate contrast controls first-step size; later differences
are consequences of diverging trajectories, not size-matched interventions at
all steps. Per-layer matching does not match every parameter, neuron or head.

Exact validation-gradient projections, history/current projections, actual
update norms and layerwise projections are measured for the first branch step.
Test losses and both accuracy definitions are evaluated but never select events
or interventions. The test-gradient/path audits, when enabled, are offline.

## Kept fixed from the seed-11 study

- Model revision: EleutherAI/pythia-410m,
  `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`.
- Dataset revisions: SST-2 `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`;
  AG News `eb185aade064a813bc0b7f42de02595523103ca4`.
- A acquisition: 600 steps. B: 128 steps. LR 2e-5. Adam betas .9/.99.
  Clip 1, microbatch 2, accumulation 2. FP32 model, FP64 loss/reductions.
- Per-task train/validation/test sizes: 1024/32/128. Maximum prefix 256 tokens.
- Frozen OLMo predictor and threshold artifact: byte-for-byte unchanged.
- One-update-ahead actions use pre-update t-1 measurements, with warm-up through
  update 9; at most 16 actions, each using 0.25x LR for that step.
- Frames: layers 0/1; calibration 16 validation examples per task; 16 held-out
  test examples per task; four random maps of each control family.

Only the seed changes the deterministic data sample/order and training minibatch
sequence; the pretrained weights are the same pinned checkpoint. These replications
therefore include variation in the sampled examples, not solely optimizer randomness.
The original OLMo source used test-population gradients; the new operational
policies use validation gradients, as in the seed-11 reviewer run. Disclose that
measurement shift. Raw projection ranking does not require fitting a classifier.

## Outputs

- `REPORT.txt`, `summary.json`: final/mean/peak A and B losses, acquisition quality,
  restricted and full-vocabulary accuracy, positive excess and loss decomposition.
- `prediction_metrics.json`: unchanged source-only baselines, AUROC/AP/Brier where
  applicable, leads 0/1 and loss-event thresholds .02/.05/.10.
- `policy_random_comparison.json`: policy effects against the five random outcomes
  and their mean, separately by seed. Random schedules are not extra model seeds;
  no significance claim is inferred from five schedules.
- `event_history_summary.json`: immediate and continuation effects, first-step
  projections and actual norm-matching audits, with unfinished branches labelled.
- `fixed_acquisition_targets.json`: retention at B-validation losses .75, 1, 1.5,
  and 2, requiring either one or four consecutive qualifying endpoints. Targets
  are fixed before these fresh runs; unreached targets remain missing. These are
  different diagnostic slices, not a license to choose the best-looking one.
- `frame_summary.json`: endpoint, baseline, forgetting and interaction effects.
- `curvature_reaudit.json`: every historical re-audit width, passing and failing.
- `results.sqlite`: all declared prompt-level records and measurement metadata.

For old-task targets q supported on answer set S, with M=sum(p_j, j in S):

    KL(q || p_full) = KL(q || p_S/M) - log(M)
                        confusion    leakage

One-hot natural-text targets give the equivalent NLL identity. Restricted-answer
accuracy can conceal probability moving outside the sentiment answer set A/B,
including the news labels C/D. The decomposition does not identify the destination
of leaked mass beyond membership outside S.

## Historical curvature re-audit

The notebook searches the known previous server directories and enables this
stage if it finds the completed seed-11 anchor, exact data and SQLite records.
Otherwise it explicitly reports that the optional historical audit is disabled;
new-seed work still runs. You can set `curvature_reaudit_source` to the full run
path before launch. Old artifacts are never modified.

For the original update's actual displacement, it measures the exact directional
Hessian and central differences at alpha .30, .328125, .36 and widths .04, .02,
.01, .005, .0025, .00125. All results are retained. The primary pass criterion
requires both of the smallest two widths to agree with autograd, with each other,
and to pass slope checks. Other adjacent-width checks are descriptive. A completed
re-audit may still fail; no width is chosen afterward to manufacture a pass.

Deferred path `resolved` concerns integration and slope checks. Always inspect
`hessian_audit.passed` separately; a resolved integral does not validate curvature.

## Running and resuming

Use the existing compatible CUDA environment. The notebook prefers
`/home/ubuntu/4/env_qwen3/bin/python` when present and checks imports there.
Additional requirements are supplied; no package installation is automatic.
Stop is cooperative. Refresh until `alive=False`, then Launch/resume continues.
Restart creates a fresh output directory and never deletes prior results.
Scientific config or code changes require a fresh directory. Hours and storage
limits can change on resume. Do not change environment versions mid-run.

The worker writes one shared A anchor per seed and one rolling checkpoint for
an active trajectory/event branch. Atomic saving temporarily needs two copies.
Default reserve: 40 GiB free filesystem space; output cap: 100 GiB. No raw gradient
or activation arrays are archived. Completed prompt records, statistics, pinned
data, deterministic batch recipe and restart states are retained; arbitrary
new tensor analyses require replay. Normal trajectories can replay an unfinished
8-update chunk; event branches checkpoint every 2 updates.

`study.show_results(SETTINGS)` prints numbers and shows A/B trajectories.
`study.export(SETTINGS)` creates `overnight_replication_share.zip` without model
checkpoints. Keep every file in this code folder together.

Dataset documentation:
https://huggingface.co/datasets/stanfordnlp/sst2
https://huggingface.co/datasets/fancyzhx/ag_news
