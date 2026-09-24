# Performance overlay validation

Local validation on a tiny original-OLMo CPU fixture; these are implementation checks, not forgetting research results or GH200 timing guarantees.

Passed:

- Every captured array, local derivative and evaluation metric matches the original capture on all fixture prompts.
- Full optimized smoke campaign completes 5/5 jobs. Every factorial corner mean and interaction matches the original implementation within 1e-10.
- Original, first-moment-reset and quarter-learning-rate continuations reproduce original records.
- Interrupted recovery replays updates after a rolling checkpoint and reproduces the uninterrupted records.
- Simulated low disk space skips an optional optimizer snapshot; recovery from the original source still reproduces the recorded reset-first-moment trajectory.
- Existing signed campaign resumes in the same directory; completed-job markers remain unchanged.
- Completed optimizer cleanup retains raw measurements.
- Unchanged source hashes are reused after initial verification.
- Byte-identical NPZ files can be hard-linked without changing their contents. Atomic rewriting of one linked path preserves the other path's data.
- Staging cleanup removes only targeted uncommitted optimizer/ZIP files and preserves committed files.
- Python syntax and notebook schema/code cells pass validation.

The original three Python modules are not modified by this add-on. Scientific and performance provenance are recorded separately. Full-scale GPU speed, full-queue storage requirements, and numerical behavior on the user's GH200 remain to be measured. A per-engine capture equivalence audit runs on the actual machine before accepting newly captured data.
