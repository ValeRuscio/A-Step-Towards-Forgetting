# Resume the existing mechanism-bridge campaign with less overhead

This is an **add-on**, not a replacement experiment. Put `bridge_fast.py` beside your existing `mechanism_bridge.py`, `bridge_core.py`, and `bridge_replay.py`. Keep those original three files unchanged: their signature protects your existing results.

Open the included `Apply_speedup.ipynb`, or run this in your current notebook:

```python
import bridge_fast as fast
SETTINGS = fast.resume_existing(
    "/home/ubuntu/1/runs/olmo_mechanism_bridge_v1", hours=11.5
)
study = fast
if "remember" in globals():
    remember()
```

This stops the verified current worker, loads the exact saved settings, and resumes the **same directory**. It does not reset the 14 completed jobs. Incomplete units continue or replay as needed. If the original signature differs, the add-on refuses to mix results instead of creating a new campaign. Repeated launch remains idempotent. Resuming grants a new session of the requested duration. An existing worker must stop to load the add-on; uploading the file alone does not change a running process.

Refresh with:

```python
_ = fast.refresh(SETTINGS["output"])
```

Your existing report/export and stop/resume controls remain available. Status now distinguishes `disk_paused` from `budget_paused`.

## What is optimized

1. **One capture forward instead of three.** Attention operands, activation derivatives, final representations, and loss metrics come from the same model forward. Both backward derivative computations remain. Every layer, token position, entity, intervention, and recorded quantity requested in your saved settings remains. The first new captured prompt in each engine is checked against the original implementation before accepting the fast path. Reconstruction and finite-value audits stay active.

2. **Completed optimizer snapshots are cleaned up automatically.** The worker removes `resume.pt` only from branches with a DONE marker and a complete curve. Source checkpoints, recorded measurements, raw NPZ arrays, and incomplete-branch snapshots are kept. Completed model/optimizer snapshots are derived restart conveniences: they can be reconstructed from the original source and branch recipe. You lose immediate direct loading of those finished snapshots, not the recorded scientific measurements. This is the same cleanup you previously performed manually. An audit log records removals.

3. **Fewer checkpoint writes.** Model+Adam restart snapshots are written every four steps rather than every step. All step measurements are committed at every step. Recovery deterministically replays any intervening updates and checks the recovered losses against the saved curve. When space is insufficient for a convenience snapshot, the code retains the measurements and relies on replay rather than stopping solely for that snapshot. The general free-space reserve still applies.

4. **Correct additional-space accounting.** The checkpoint check budgets one new snapshot plus margin/reserve; it no longer requires room for two additional snapshots when an existing snapshot already occupies disk. Temporary writes remain atomic. Uncommitted optimizer/ZIP temporary files left by a stopped worker are cleaned up under the exclusive worker lock. A final convenience snapshot is not written because the completed branch no longer needs it for the queue.

5. **Cached source hashes.** The first upgraded session fully verifies the source once. Later resumes reuse hashes when device, inode, size, nanosecond modification time, and change time match. Changed files are rehashed and compared against the original fingerprints. This assumes trustworthy filesystem metadata. You can require full hashing every time through the runtime options below.

6. **Less report overhead.** Automatic reports are rebuilt every four completed jobs and at pause/completion. You can refresh them manually at any time. Status no longer traverses over 100,000 files just to count diagnostic records.

No reduced precision, altered attention backend, reduced sampling, fewer layers, fewer gradients, or changed learning rules are introduced. The original scientific settings/signature stay intact; add-on version, code hash and runtime settings are logged separately in `performance_history.json`.

## Disk space is still finite

This does not make an expanding raw-data collection constant-sized. Keeping all full-resolution activations and derivatives across all 96 jobs may need substantially more space than the earlier 150–200 GB guidance, which was too optimistic for the complete raw-capture queue. Measure your actual usage rather than treat that estimate as a guaranteed bound.

Use this optional diagnostic (it scans the output tree once):

```python
_ = fast.storage_report(SETTINGS["output"])
```

The completed-optimizer cleanup limits redundant snapshot accumulation. If raw arrays become the dominant storage cost, more disk or external archival storage is needed to retain everything.

## Optional genuinely lossless compaction

The add-on can replace byte-identical NPZ files with hard links. Every filename and array remains readable; no array precision or contents change. It verifies byte equality, operates only inside the output directory while the worker is stopped, and does not follow file symlinks. Subsequent atomic writes break the link safely. Savings depend on the amount of exact duplication and filesystem hard-link support; this is not guaranteed to save much.

```python
print(fast.stop(SETTINGS["output"]))
_ = fast.deduplicate(SETTINGS["output"])
OUTPUT = fast.launch(SETTINGS)
```

Deduplication reads candidate files and may take time. It is optional and not repeatedly run in the background. Copying these files to storage that does not preserve hard links can expand their physical size again. Exports retain their logical contents.

## Runtime options

The worker reads `performance_options.json` separately from scientific settings. Defaults:

```json
{
  "checkpoint_every": 4,
  "cleanup_completed_checkpoints": true,
  "report_every_jobs": 4,
  "verify_all_source_hashes": false
}
```

Disabling cleanup preserves existing convenience snapshots, but the optimized worker still does not promise a final optimizer snapshot for every completed branch. Use the original worker if retaining every final optimizer snapshot is a requirement. Never delete source checkpoints: replay depends on them.

## Validation and expected speed

Tested on a tiny original-OLMo CPU fixture:

- Fused arrays, local derivatives, and loss metrics match the original capture on every fixture prompt.
- All three continuation variants match their original recorded trajectories.
- Interrupted continuation recovery after a rolling checkpoint reproduces the original records.
- An original signed run upgrades in place and retains DONE markers.
- Completed-checkpoint cleanup preserves measurements.
- Source hash reuse works for unchanged files.
- NPZ deduplication preserves byte contents; subsequent atomic replacement leaves other linked files unchanged.

Full-scale GH200 speed has not been measured. Removing two capture forwards does not imply the entire campaign becomes three times faster: backward passes, intervention evaluations, geometry, hashing, storage, and replay still contribute. Job counts also have unequal costs, so 14/96 after eight hours is not a reliable linear ETA.
