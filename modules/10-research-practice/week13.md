# Week 13 — Reproducible Machine-Learning Research and Peer Review

**Guiding question:** What evidence allows another researcher to trace, rerun, and critically evaluate a machine-learning result?

Week 13 turns the semester project into a reviewable research artifact. Students connect every claim to a result, every result to a configuration and command, and every limitation to a feasible next experiment. The workshop treats independent reproduction as an empirical test of the research package.

## Learning outcomes

By the end of the session, students can:

- separate an observation, statistical summary, inference, and broader claim;
- use repeatability, reproducibility, and replicability with declared definitions;
- identify the code, configuration, data identity, environment, and command behind a result;
- calculate a sample mean, standard deviation, paired difference, and confidence interval;
- distinguish file integrity from numerical agreement and scientific validity;
- specify a tolerance before attempting reproduction;
- classify construct, internal, external, and statistical threats to validity;
- reproduce a peer result from a fixed repository state;
- write a focused review comment that another researcher can act on;
- revise a draft report so its claims match its evidence;
- turn one unresolved limitation into a master's or PhD research question.

## Preparation

Read:

- [Improving Reproducibility in Machine Learning Research](https://www.jmlr.org/papers/v22/20-303.html);
- the [ACM artifact review and badging policy](https://www.acm.org/publications/policies/artifact-review-and-badging-current);
- the [NeurIPS Paper Checklist guidelines](https://neurips.cc/public/guides/PaperChecklist).

Bring a project repository at a fixed commit. It should contain one command that reproduces one table row or figure, a dependency file, the configuration actually used, and an expected result with a stated tolerance. Remove credentials, private records, and restricted data before exchanging artifacts.

## Four-hour teaching sequence

| Time | Activity | Evidence |
| ---: | --- | --- |
| 0–20 min | Claim-to-evidence audit | One project claim is decomposed into observation and inference |
| 20–45 min | Reproducibility terminology | Team, artifacts, and experimental setup are identified |
| 45–75 min | Artifact anatomy | Commit, environment, data identity, command, and output are traced |
| 75–105 min | Repeated-run uncertainty | Mean, sample standard deviation, pairing, and interval are calculated |
| 105–120 min | Reproduction tolerance | Exact, numerical, and behavioral agreement are distinguished |
| 120–130 min | Break | — |
| 130–165 min | Lab 13 audit | A deterministic bundle is produced, hashed, and verified |
| 165–205 min | Peer reproduction | A partner follows written instructions without verbal repair |
| 205–225 min | Threats to validity | One threat and one bounded consequence are written |
| 225–245 min | Review and revision | One blocking issue becomes a verified documentation fix |
| 245–260 min | Draft report and research direction | Claims, limitations, and next experiment are aligned |

## The claim-to-evidence chain

A research result should permit this trace:

$$
\text{claim}
\longleftarrow
\text{analysis}
\longleftarrow
\text{recorded outputs}
\longleftarrow
\text{configuration and code}
\longleftarrow
\text{data and environment}.
$$

A missing link limits the claim. A score without a split identity cannot establish performance on a declared population. A figure without a generating command cannot be audited. A successful rerun confirms a computational path, while the research design still requires separate evaluation.

Use four labels when drafting:

| Level | Example | Required support |
| --- | --- | --- |
| observation | Test accuracy was 0.634 for seed 13 | output tied to a command and configuration |
| summary | Mean accuracy across five seeds was 0.634 | individual runs and summary method |
| comparison | The interaction increased accuracy by 0.113 on average | paired baseline and treatment runs plus uncertainty |
| scope claim | The interaction helps similar nonlinear tasks | external evidence or a clearly limited hypothesis |

## Reproducibility terminology

Terminology varies across disciplines. This course follows the ACM framing and always states the actors and artifacts:

| Term | Team | Experimental setup | Course example |
| --- | --- | --- | --- |
| repeatability | same team | same artifacts and setup | rerun the saved command |
| reproducibility | different team | author artifacts and declared setup | a peer reruns the fixed commit |
| replicability | different team | independently created artifacts or setup | a peer reimplements the method |

The label alone is insufficient. Record who ran the experiment, which artifacts they used, what changed, and whether the result agreed within a prespecified tolerance.

## Artifact anatomy

A minimal machine-learning result package identifies:

1. **Repository state:** immutable commit, release, or archive checksum.
2. **Entry command:** one command from the repository root.
3. **Environment:** supported Python version and resolved package versions.
4. **Data identity:** source, version, license, checksum or immutable identifier, exclusions, and split files.
5. **Configuration:** model, preprocessing, optimizer, selection rule, seeds, and compute budget.
6. **Output contract:** expected files, metrics, units, split roles, and tolerance.
7. **Provenance:** the command and configuration that generated every reported table or figure.
8. **Limitations:** unavailable assets, nondeterministic operations, cost, and known scope restrictions.

A manifest makes integrity testable:

$$
H_i=\operatorname{SHA256}(\text{file}_i).
$$

Matching $H_i$ verifies bytes. It does not establish that labels are correct, the metric suits the claim, the test set stayed locked, or the model will generalize.

## Repeated runs and paired comparisons

For measurements $x_1,\ldots,x_n$, report

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,
$$

and the sample standard deviation

$$
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}.
$$

When two conditions use the same seeds and data partitions, analyze paired differences

$$
d_i=x_i^{(B)}-x_i^{(A)}.
$$

A two-sided 95% Student-$t$ interval for the mean paired difference is

$$
\bar{d}\pm t_{0.975,n-1}\frac{s_d}{\sqrt{n}}.
$$

State what varies across runs. Five training seeds do not measure variation across hospitals, years, devices, annotators, or dataset construction choices.

## Agreement criteria

Choose the criterion before the reproduction attempt:

| Criterion | Appropriate use | Example |
| --- | --- | --- |
| exact bytes | immutable source, configuration, split files | SHA-256 hashes match |
| numerical tolerance | deterministic calculations with small platform differences | absolute metric difference at most $10^{-6}$ |
| interval overlap or directional agreement | stochastic training or hardware variation | improvement remains positive across paired seeds |
| qualitative behavior | systems where absolute time depends on hardware | method B remains faster under the declared setup |

A changed result can expose incomplete instructions, dependency drift, hidden state, nondeterminism, hardware sensitivity, or a scientific assumption. Record the discrepancy before trying to repair it.

## Lab 13 — artifact production and independent verification

Run the no-file audit:

```bash
python labs/week13_reproducibility_audit.py --audit-only
```

Produce and immediately verify the reference bundle:

```bash
python labs/week13_reproducibility_audit.py
```

The script writes these ignored outputs beneath `outputs/week13/reference/`:

- `config.json` with the full controlled experiment;
- `environment.json` with versions, platform, command, and commit;
- `runs.csv` with paired seed-level results;
- `summary.json` with means, standard deviations, and a paired interval;
- `manifest.json` with schema, numeric tolerance, source hash, and file hashes.

Verify an existing bundle independently:

```bash
python labs/week13_reproducibility_audit.py --verify outputs/week13/reference
```

The generated task compares a linear baseline with one prespecified interaction feature. Data, split, optimization budget, seed set, selection rule, and evaluation stay fixed. The default result is a teaching example, not evidence about a real application.

## Threats to validity

| Threat | Question | Example response |
| --- | --- | --- |
| construct | Does the measurement represent the intended concept? | Accuracy may hide rare-class failure |
| internal | Could another changed factor explain the result? | Preprocessing differed between conditions |
| statistical conclusion | Is the comparison precise and analyzed correctly? | Five seeds give a wide interval |
| external | Where should the conclusion generalize? | One dataset does not support a cross-domain claim |

A limitation states the affected claim and likely consequence. “More data are needed” is incomplete. Prefer: “The test set contains one device type, so the result does not establish robustness to a new sensor; evaluate a device-held-out split next.”

## Peer-reproduction protocol

The author supplies a fixed commit and written instructions. The reviewer works without verbal guidance for the first attempt and records:

- environment creation and actual versions;
- exact commands and exit status;
- expected and observed artifacts;
- agreement criterion and result;
- deviations from the instructions;
- the first blocking ambiguity;
- the smallest documentation or code change that resolves it;
- the verification command after the correction.

The reviewer evaluates the claim and evidence separately from the ease of setup. A failed reproduction can produce strong work when its diagnosis is precise and the follow-up test isolates the cause.

## Review comments

A useful comment contains four parts:

1. **Location:** file, section, command, or result.
2. **Observation:** what happened, including relevant output.
3. **Consequence:** which reproduction step or claim is affected.
4. **Requested change:** the smallest verifiable correction.

Example:

> In `README.md`, the training command omits the split configuration. Running it at commit `abc123` uses the default random split and gives 0.71 rather than the expected 0.78. Please add the exact config path and expected tolerance; I will rerun the corrected command.

## Draft-report audit

Use the [report template](../../research-project/report-template.md). For every table and figure, provide its data split, units, uncertainty, and generating command. Keep published results separate from student results. The abstract should state the problem, method, main evidence, and bounded conclusion without adding claims that appear nowhere else.

Complete the expanded [reproducibility review assignment](../../assignments/05-reproducibility-review.md). Submit the review record together with the revised draft report.

## Research direction

Turn the strongest unresolved limitation into a next experiment:

$$
\text{observed limitation}
+\text{candidate mechanism}
+\text{discriminating experiment}
+\text{decision rule}.
$$

A research direction becomes credible when the outcome could change the current explanation. State the data, expertise, compute, access, or ethical review required before proposing it as master's or PhD work.

## Package

- [Detailed notes](week13-notes.md)
- [Accessible slide source](week13-slides.md)
- [Student worksheet](week13-worksheet.md)
- [PowerPoint deck](slides/week13-reproducible-research-and-peer-review.pptx)
- [Instructor guide](../../instructor-notes/week13.md)
- [Reproducibility-audit lab](../../labs/week13_reproducibility_audit.py)
- [Further reading](../../readings/week13-reproducibility-and-peer-review.md)
- [Reproducibility review assignment](../../assignments/05-reproducibility-review.md)
- [Report template](../../research-project/report-template.md)

## Exit ticket

In four sentences:

- distinguish reproduction from scientific validation;
- name the fixed repository, data, and environment identifiers;
- state the agreement criterion for one result;
- convert one limitation into a discriminating next experiment.