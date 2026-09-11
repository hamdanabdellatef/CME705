# Week 13 worksheet — Reproduction, Review, and Research Direction

**Name:** ______________________________  **Project:** ______________________________

Use this worksheet during the claim audit, lab, and peer-reproduction workshop. Record observed evidence before explanations or repairs.

## 1. Decompose one project claim

Copy one sentence from the draft report:

> ____________________________________________________________________________

Classify it:

- [ ] observation from one run
- [ ] summary across runs
- [ ] controlled comparison
- [ ] causal or mechanistic inference
- [ ] population or generalization claim

Complete the trace:

| Link | Exact project evidence |
| --- | --- |
| claim | |
| analysis or calculation | |
| output file and field | |
| generating command | |
| configuration | |
| code commit | |
| dataset and split identity | |
| environment and compute | |

Which missing link most limits the claim?

> ____________________________________________________________________________

Rewrite the claim at the scope supported by the evidence:

> ____________________________________________________________________________

## 2. Terminology cases

This course follows the ACM framing. Classify each case and explain the actors and artifacts.

### Case A

The original student reruns the same commit and command on the same laptop.

Term: __________________________

Reason: ______________________________________________________________________

### Case B

A classmate creates a fresh environment, checks out the author's fixed commit, and runs the supplied command.

Term: __________________________

Reason: ______________________________________________________________________

### Case C

A different team reimplements the method from the report and evaluates it on an independently collected dataset.

Term: __________________________

Reason: ______________________________________________________________________

Why should a report describe the team, artifacts, and setup even after choosing a term?

> ____________________________________________________________________________

## 3. Artifact inventory

For your project, record:

| Required item | Location or status | Correction needed |
| --- | --- | --- |
| immutable commit or archive checksum | | |
| setup command | | |
| entry command | | |
| machine-readable configuration | | |
| resolved environment versions | | |
| data version or checksum | | |
| split IDs or generation procedure | | |
| expected output path | | |
| expected result | | |
| agreement tolerance | | |
| compute estimate | | |
| known restrictions or limitations | | |

## 4. Integrity and validity

A manifest reports that every file hash matches.

What does this establish?

> ____________________________________________________________________________

Name four conclusions it does not establish:

1. ___________________________________________________________________________
2. ___________________________________________________________________________
3. ___________________________________________________________________________
4. ___________________________________________________________________________

## 5. Calculate repeated-run summaries

The Week 13 lab records paired accuracy differences:

$$
[0.1111,\ 0.1222,\ 0.1111,\ 0.1333,\ 0.0889].
$$

### 5.1 Mean difference

$$
\bar{d}=\frac{1}{n}\sum_{i=1}^{n}d_i.
$$

Calculation:

> ____________________________________________________________________________

Result: __________________________

### 5.2 Sample standard deviation

$$
 s_d=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(d_i-\bar{d})^2}.
$$

Result: __________________________

### 5.3 Standard error

$$
\operatorname{SE}(\bar{d})=\frac{s_d}{\sqrt{n}}.
$$

Result: __________________________

### 5.4 Confidence interval

Use $t_{0.975,4}=2.776$:

$$
\bar{d}\pm2.776\operatorname{SE}(\bar{d}).
$$

Interval: [__________________, __________________]

### 5.5 Interpretation

What source of variation does this interval represent?

> ____________________________________________________________________________

What important sources of uncertainty are absent?

> ____________________________________________________________________________

## 6. Choose an agreement criterion

For each artifact, select exact bytes, numerical tolerance, behavioral agreement, or statistical agreement.

| Target | Criterion | Threshold or decision rule | Why appropriate? |
| --- | --- | --- | --- |
| source file | | | |
| split identifiers | | | |
| validation loss | | | |
| mean test metric | | | |
| inference time on different GPU | | | |
| ordering of two methods | | | |

State the criterion for the peer reproduction before running it:

> ____________________________________________________________________________

## 7. Lab 13 audit

Run:

```bash
python labs/week13_reproducibility_audit.py --audit-only
```

Record:

| Check | Observed value |
| --- | --- |
| canonical hash stable | |
| number of run records | |
| paired-run count | |
| split sizes | |
| partitions disjoint | |
| files written | |
| downloads started | |

Which checks concern integrity? Which concern experimental design?

> ____________________________________________________________________________

## 8. Produce the reference bundle

Run:

```bash
python labs/week13_reproducibility_audit.py
```

Complete:

| Record | Value |
| --- | --- |
| artifact directory | |
| Git commit | |
| files covered by manifest | |
| source SHA-256 | |
| baseline mean accuracy | |
| interaction mean accuracy | |
| mean paired difference | |
| 95% paired interval | |
| verification status | |

Explain why the test partition stays locked until the configuration and selection procedure are fixed:

> ____________________________________________________________________________

## 9. Inspect the manifest

Open `outputs/week13/reference/manifest.json`.

- Schema: ____________________________________________________________________
- Numeric tolerance: __________________________________________________________
- Hashed files: _______________________________________________________________
- Source path: ________________________________________________________________

If one byte in `runs.csv` changes, what should verification report?

> ____________________________________________________________________________

Why should the manifest exclude secrets and private machine paths?

> ____________________________________________________________________________

## 10. Peer reproduction record

### Identity

- Author: ____________________________________________________________________
- Reviewer: __________________________________________________________________
- Repository: _________________________________________________________________
- Author commit: ______________________________________________________________
- Review date: ________________________________________________________________

### Environment

| Item | Author declaration | Reviewer observation |
| --- | --- | --- |
| operating system | | |
| Python | | |
| NumPy/PyTorch | | |
| accelerator | | |
| installation command | | |

### First attempt

- Exact command: ______________________________________________________________
- Exit status: ________________________________________________________________
- Expected output: ____________________________________________________________
- Observed output: ____________________________________________________________
- Expected result and tolerance: ______________________________________________
- Observed result: ____________________________________________________________
- Time and compute: ___________________________________________________________

Status:

- [ ] reproduced
- [ ] partial
- [ ] inconclusive
- [ ] different
- [ ] blocked

Evidence for the status:

> ____________________________________________________________________________

## 11. Diagnose before repairing

First blocking or material discrepancy:

> ____________________________________________________________________________

Classify its likely source:

- [ ] missing instruction
- [ ] dependency drift
- [ ] data or split mismatch
- [ ] configuration mismatch
- [ ] hidden state
- [ ] nondeterminism
- [ ] hardware sensitivity
- [ ] analysis or metric mismatch
- [ ] possible implementation error
- [ ] resource or access limit

Evidence supporting this diagnosis:

> ____________________________________________________________________________

Smallest test that could distinguish this explanation from another:

> ____________________________________________________________________________

## 12. Write one focused review comment

**Location:** __________________________________________________________________

**Observed evidence:**

> ____________________________________________________________________________

**Consequence for reproduction or claim:**

> ____________________________________________________________________________

**Requested change:**

> ____________________________________________________________________________

**Verification plan:**

> ____________________________________________________________________________

## 13. Revision verification

- Corrected commit: ___________________________________________________________
- Command rerun: ______________________________________________________________
- New result: __________________________________________________________________
- Status after correction: ____________________________________________________
- Remaining limitation: _______________________________________________________

## 14. Threats to validity

Write one threat in each category.

| Category | Specific project threat | Affected claim | Likely consequence | Next check |
| --- | --- | --- | --- | --- |
| construct | | | | |
| internal | | | | |
| statistical conclusion | | | | |
| external | | | | |

Rewrite the strongest threat as a bounded limitation:

> Because ____________________________________________________________________,
> the result does not establish _______________________________________________.
> This may ___________________________________________________________________.
> Test _______________________________________________________________________.

## 15. Draft-report audit

For each item, record a page, table, figure, file, or missing status.

| Report component | Evidence location | Revision needed |
| --- | --- | --- |
| bounded research question | | |
| related-work comparison | | |
| data provenance and license | | |
| split and preprocessing protocol | | |
| baseline | | |
| method and controlled factor | | |
| selection rule | | |
| repeated-run uncertainty | | |
| locked-test result | | |
| error or subgroup evidence | | |
| limitations | | |
| generating command for each result | | |
| reproducibility statement | | |

## 16. Research direction

Complete:

- Observed limitation: ________________________________________________________
- Candidate mechanism: ________________________________________________________
- Competing explanation: ______________________________________________________
- Discriminating experiment: __________________________________________________
- Decision metric and threshold: ______________________________________________
- Required data or access: ____________________________________________________
- Required skills and compute: ________________________________________________
- Ethical or institutional review: ____________________________________________

What result would change the current conclusion?

> ____________________________________________________________________________

Why could this become a credible master's or PhD direction?

> ____________________________________________________________________________

## Exit ticket

1. A successful reproduction establishes ______________________________________
2. It does not establish ______________________________________________________
3. Our agreement criterion was ________________________________________________
4. The next experiment is useful because ______________________________________