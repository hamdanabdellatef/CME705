# CME705 Research Project Report Template

Target length: approximately 5–7 pages excluding references and appendices. Replace prompts with project content. Remove sections that do not apply only when the omission is explained.

## Title

Name the task, studied mechanism or method, and evaluation setting. Avoid claiming a general result in the title when the evidence comes from one dataset.

**Title:** ____________________________________________________________________

**Author:** ______________________________  **Repository:** ______________________

**Final commit or release:** ___________________________________________________

## Abstract

Write 150–200 words covering:

- problem and intended setting;
- data and evaluation protocol;
- baseline and main method or controlled factor;
- main result with metric and uncertainty;
- bounded conclusion and strongest limitation.

Do not introduce claims or experiments that the report does not support.

> ____________________________________________________________________________

## 1. Problem and research question

### 1.1 Application and unit of observation

State what one observation represents, who or what the population contains, the prediction or representation target, and the intended use.

### 1.2 Research question

Use a form such as:

> Under [data and evaluation setting], how does [one factor] affect [metric or behavior] relative to [baseline], while [controls] remain fixed?

### 1.3 Contribution and scope

List the concrete contribution. Bound it by dataset, task, metric, and study design.

## 2. Related work

Compare at least five credible sources.

| Source | Task and data | Method | Evaluation | Main evidence | Limitation or difference from this project |
| --- | --- | --- | --- | --- | --- |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

State the gap or unresolved question that motivates this experiment. Do not define novelty as “no one used model X on dataset Y” without explaining the scientific or practical contribution.

## 3. Data

### 3.1 Provenance and access

| Item | Value |
| --- | --- |
| dataset and version | |
| source and access date | |
| license or terms | |
| unit of observation | |
| population and exclusions | |
| labels or targets | |
| known collection limitations | |

### 3.2 Split protocol

Describe train, validation, and test construction. State group, temporal, geographic, subject, or device separation. Provide split identifiers, a deterministic generator, or a checksum.

### 3.3 Preprocessing

Identify every fitted transform and confirm it was fitted using training data only. Describe missing-data handling, normalization, tokenization, augmentation, and feature construction.

## 4. Methods

### 4.1 Baseline

Explain why the baseline is credible and what comparison it anchors.

### 4.2 Main method

Describe inputs, outputs, architecture or algorithm, objective, regularization, and inference procedure. Give enough detail to understand and reproduce the method.

### 4.3 Controlled factor

| Decision | Baseline | Treatment | Fixed? |
| --- | --- | --- | --- |
| data and split | | | yes |
| preprocessing | | | |
| architecture or representation | | | |
| optimization budget | | | |
| selection rule | | | yes |
| seed set | | | yes |
| metric implementation | | | yes |

State what changed and why that change tests the hypothesis.

## 5. Experimental protocol

### 5.1 Configuration

Record optimizer, learning rate, batch size, epochs, stopping rule, checkpoint selection, seeds, precision, hardware, and compute budget.

### 5.2 Model selection

State which validation quantity selected the final configuration. Record how many candidates were evaluated. Explain all test-data use.

### 5.3 Metrics and uncertainty

Define each metric, its direction, averaging rule, units, and implementation. State what varies across repeated runs and how uncertainty is calculated.

For paired runs:

$$
d_i=x_i^{(B)}-x_i^{(A)}.
$$

Report $\bar{d}$, $s_d$, and the interval or decision rule chosen before examining the final result.

## 6. Results

### 6.1 Main result

| Condition | Validation metric | Locked-test metric | Variation or interval | Seeds | Compute |
| --- | ---: | ---: | --- | --- | --- |
| baseline | | | | | |
| treatment | | | | | |

Generating command: ___________________________________________________________

Result file: __________________________________________________________________

### 6.2 Controlled comparison

State the paired or otherwise matched effect. Explain whether it supports the directional prediction.

### 6.3 Error or subgroup evidence

| Observation or subgroup | Target | Prediction | Context | Observed pattern | Possible explanation |
| --- | --- | --- | --- | --- | --- |
| | | | | | |
| | | | | | |
| | | | | | |

Separate observed facts from hypotheses.

### 6.4 Robustness check

Report one stress test aligned with a plausible failure mode. Keep the original test result unchanged and label any post-test analysis as exploratory.

## 7. Discussion

### 7.1 Supported claim

State the strongest conclusion supported by the design and evidence.

### 7.2 Comparison with prior work

Separate published results from project-generated results. Discuss protocol differences before comparing numbers.

### 7.3 Alternative explanations

Name at least one explanation that the current experiment cannot distinguish.

## 8. Limitations and threats to validity

| Category | Specific threat | Affected claim | Likely consequence | Next check |
| --- | --- | --- | --- | --- |
| construct | | | | |
| internal | | | | |
| statistical conclusion | | | | |
| external | | | | |

Use this structure:

> Because [evidence boundary], the result does not establish [affected claim]. This may [likely consequence]. Test [specific next experiment].

Include data access, licensing, privacy, fairness, security, misuse, or human-subject constraints when relevant.

## 9. Conclusion

Answer the research question using the result and its scope. Do not repeat the abstract or add new claims.

## 10. Research direction

- observed limitation: _________________________________________________________
- candidate mechanism: _________________________________________________________
- competing explanation: ______________________________________________________
- discriminating experiment: __________________________________________________
- decision metric and threshold: _______________________________________________
- required data, skills, compute, or approvals: _________________________________
- result that would change the current conclusion: ______________________________

Explain whether and how this could develop into master's or PhD research.

## 11. Reproducibility statement

| Item | Value |
| --- | --- |
| repository URL | |
| commit or release | |
| supported environment | |
| installation command | |
| dataset identity and access | |
| split identity | |
| main configuration | |
| seed set | |
| reproduction command | |
| expected output | |
| expected result and tolerance | |
| runtime, memory, and device | |
| known nondeterminism | |

State which artifacts are public, restricted, unavailable, or omitted and why.

## References

Use one consistent citation style. Cite original sources for datasets, code, models, metrics, and methods. Record versions and access dates where relevant.

## Appendix A. Result provenance

| Report item | Source or command | Output file | Configuration | Commit |
| --- | --- | --- | --- | --- |
| Table 1 | | | | |
| Figure 1 | | | | |
| Main test result | | | | |

## Appendix B. Reproduction review

Link the completed Week 13 reproduction record, reviewed commit, correction, and verification status.

## Final author check

- [ ] Claims in the title, abstract, results, and conclusion agree.
- [ ] The data population, split roles, and intended use are explicit.
- [ ] The baseline and controlled factor are justified.
- [ ] Model selection uses validation data and final evaluation is labeled honestly.
- [ ] Variation and uncertainty are defined.
- [ ] Published and project-generated results are separate.
- [ ] Every table and figure has provenance.
- [ ] Limitations identify affected claims and consequences.
- [ ] The reproduction command starts from the repository root.
- [ ] The final commit and expected result are recorded.
- [ ] Licenses, privacy, consent, and release restrictions are addressed.
- [ ] The next experiment can change the current conclusion.