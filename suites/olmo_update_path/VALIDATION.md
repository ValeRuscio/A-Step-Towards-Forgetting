# Validation performed

Test environment: local CPU, Python 3.9, PyTorch 2.8, Transformers 4.57.6. The test source uses a tiny original OLMo architecture with four entities, two layers and two attention heads. Actual OLMo-1B GPU execution and memory consumption have not been tested here.

The final code completed a detached two-update replay (steps 8 and 9) with three path fractions per update, all four task/prompt splits, both blocks and final hidden state, topology/permutation controls and attention audits.

Passed checks:

- Independent replay without probes matches every recorded post-update A/B validation/test loss at absolute/relative tolerance 1e-7, including the second update after the first path's probes.
- Path endpoints agree with independently replayed endpoint losses.
- Initial exact path slopes agree with the original pre-update gradient/displacement calculation to absolute tolerance 1e-7.
- All six sampled path slopes pass both finite-difference checks using the default tolerance `1e-5 + .05*abs(slope)`.
- Adaptive bisection resolves a known synthetic slope reversal to a bracket no wider than .01; it does not add spurious refinement for a monotonic synthetic curve. Tiny model updates had no observed slope reversal; no real forgetting transition was demonstrated by the smoke test.
- Every path point includes all four entities in A_valid/A_test/B_valid/B_test.
- All reconstructed attention-value outputs pass the audit; maximum relative error was approximately 8.4e-8. Head probabilities sum to one, and baseline attention divergence is zero.
- Exact final answer-margin decomposition agrees within 1e-6.
- Every JSON numeric value was finite. Undefined neighbour comparisons are null. Future-interval tables exclude the retrospective endpoint label.
- The notebook schema validates, every code cell compiles, and all saved cell outputs are empty.
- Duplicate launch preserves the active PID. Compatible resume leaves completed measurement records untouched. Restart preserves the old results and creates a fresh directory. Stop terminates the tracked process.
- The generated loss/slope and geometry figures were visually inspected. Undefined target-neighbour comparisons are explicitly labelled.

The preceding trajectory implementation also tested native Adam numerator-channel reconstruction, coarse Fisher's local KL identity and label-dependent component merging on a fixed cloud. The full code retains the mathematical self-test (`python structure_study.py self-test`) and tiny end-to-end smoke command (`python structure_study.py smoke --output /a/new/smoke/folder`). The smoke command creates synthetic test checkpoints; these are not research results.

These checks support the implementation, not a causal explanation of forgetting or generalization to other LLMs/tasks. Numerical derivative audits and attention audits remain part of every real run's output.
