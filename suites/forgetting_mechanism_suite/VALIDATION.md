# Validation performed

14 September 2026.

- Seven analytical/optimizer checks passed: whitened sensitivity spectrum and metric, exact binary conditional KL, common-bias decomposition, validation-only matching, no-positive-gain handling, tied parameter injection, and Adam numerator reconstruction (including nonempty history). Some checks contain multiple assertions.
- Final end-to-end CPU smoke run completed on a fresh random tiny OLMo: 33 finite intervention rows, 4 continuation evaluation rows, saved CSV/JSON, spectrum, all-token traces and plot.
- Full-step replay maximum margin discrepancy: 0.0.
- Relative optimizer contribution reconstruction discrepancy: 3.11459775e-05.
- Restricted spectrum status: resolved_restricted_span; 8/8 primitive derivative columns resolved. Derived-direction audits are retained in the smoke output during development.
- Detached process launch and immediate status worked. Duplicate launch was refused.
- Identical-design resume skipped the completed case without changing its completion timestamp. Changed-design resume was refused.
- Notebook JSON passed nbformat validation; every code cell compiled. Its launch/status functions were exercised against the tiny source. The notebook's production configuration was not executed.
- Saved-data analysis reproduced all 18 supplied measurement cases, including 10 updates worsening both A and B. Included example results and plot come from that actual archive reanalysis.
- Example and tiny-model plots were visually inspected.

Environment used for model tests: Python 3.9, PyTorch 2.8.0, Transformers 4.57.6, CPU on macOS. Your original measurement environment was Python 3.10, PyTorch 2.11.0+cu128, Transformers 4.46.3. The existing source checkpoint/data/optimizer conventions are preserved, but production compatibility still must pass replay and data checks on your GPU machine.

**No new 1B-parameter OLMo production experiment was run here.** Tiny-model behavior is implementation validation, not scientific evidence about forgetting. Runtime and memory on your full model remain unmeasured.
