# Week 13 slides — Reproducible Research and Peer Review

Accessible source for the Week 13 PowerPoint deck. Each section corresponds to one slide.

## Slide 1: Reproducible research and peer review

**Subtitle:** From a machine-learning claim to an independently checked artifact

CME705 Machine Learning

Week 13

## Slide 2: Session evidence

By the end of the workshop, students will have:

- traced one claim to its generating command;
- produced and verified a hashed experiment bundle;
- calculated a paired effect and confidence interval;
- completed one independent reproduction attempt;
- written and verified one focused review correction;
- revised one report claim or limitation.

## Slide 3: A number becomes evidence within a protocol

> Test accuracy: 0.81

Before interpretation, identify:

- task and unit of observation;
- dataset and split;
- baseline and selection rule;
- metric definition;
- variation across runs;
- code, configuration, and environment.

A score alone cannot show how it was produced or where it should generalize.

## Slide 4: The claim-to-evidence chain

$$
\text{claim}
\longleftarrow
\text{analysis}
\longleftarrow
\text{output}
\longleftarrow
\text{configuration and code}
\longleftarrow
\text{data and environment}
$$

Every link needs a stable identifier.

A missing link narrows the supported claim.

## Slide 5: Four levels of statement

| Level | Example | Evidence |
| --- | --- | --- |
| observation | accuracy was 0.633 for seed 13 | one traceable run |
| summary | mean accuracy was 0.634 | individual runs and calculation |
| comparison | mean paired gain was 0.113 | controlled conditions and uncertainty |
| scope claim | the feature helps similar tasks | evidence across relevant settings |

## Slide 6: Research claim audit

For one sentence in the draft:

1. Mark observation, summary, comparison, inference, or scope claim.
2. Locate the exact result file.
3. Locate the command and configuration.
4. Identify data, split, and environment.
5. Rewrite the sentence at the supported scope.

## Slide 7: Reproducibility terminology

This course follows the ACM framing.

| Term | Team | Experimental setup |
| --- | --- | --- |
| repeatability | same team | same artifacts and setup |
| reproducibility | different team | author artifacts and declared setup |
| replicability | different team | independently created artifacts or setup |

Always describe team, artifacts, and setup because disciplines use different labels.

## Slide 8: What a successful rerun establishes

Supported statements:

- the command executes in the tested environment;
- the artifact files are internally consistent;
- the target result agrees within the declared criterion;
- the report points to the correct output.

Separate checks still address data validity, leakage, metric choice, uncertainty, and generalization.

## Slide 9: Minimal artifact package

A reviewable result includes:

- immutable repository state;
- environment construction and observed versions;
- data and split identity;
- resolved configuration;
- one entry command;
- expected output and agreement criterion;
- result provenance;
- known limits and resource needs.

## Slide 10: Repository identity

A branch moves. A commit identifies one state.

```text
repository: https://example.org/project
commit: 4b11d2...
command: python train.py --config configs/final.json
output: outputs/final/metrics.json
```

Record `git status --short` before the final run.

## Slide 11: Environment identity

Construction recipe:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Observed environment:

```bash
python --version
python -m pip freeze
```

Also record operating system, compute device, precision, memory, and elapsed time.

## Slide 12: Data and split identity

Record:

- source, version, license, and access route;
- unit of observation and population;
- inclusion, exclusion, and label construction;
- training-fitted preprocessing;
- split identifiers or deterministic procedure;
- group, temporal, site, or device separation.

Restricted data need permitted access instructions or a synthetic audit path.

## Slide 13: Configuration identity

Source code shows possible settings. A result needs the settings actually used.

Capture:

- representation and model;
- loss, optimizer, schedule, and regularization;
- batch size and training budget;
- selection metric and checkpoint rule;
- seed set;
- inference and evaluation settings.

Save the resolved configuration beside the output.

## Slide 14: Randomness has several sources

Training variation can come from:

- initialization;
- sample order and augmentation;
- stochastic layers;
- parallel reduction order;
- accelerator kernels;
- library algorithms.

A seed initializes random generators. It does not guarantee identical results across every platform.

## Slide 15: Repeated-run summaries

For results $x_1,\ldots,x_n$:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

$$
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}
$$

Name what varies across runs.

Standard deviation describes run-to-run spread.

## Slide 16: Paired comparisons

When both conditions share seed $i$:

$$
d_i=x_i^{(B)}-x_i^{(A)}
$$

Pairing removes shared variation when the same seed, split, or observation has scientific meaning.

Use the same seed set for both conditions.

## Slide 17: Week 13 seed-level evidence

| Seed | Baseline accuracy | Interaction accuracy | Paired difference |
| ---: | ---: | ---: | ---: |
| 13 | 0.5222 | 0.6333 | 0.1111 |
| 29 | 0.5167 | 0.6389 | 0.1222 |
| 47 | 0.5167 | 0.6278 | 0.1111 |
| 71 | 0.5111 | 0.6444 | 0.1333 |
| 101 | 0.5389 | 0.6278 | 0.0889 |

Fixed data, split, optimizer, budget, selection rule, and metric.

## Slide 18: Paired interval calculation

$$
\bar{d}\pm t_{0.975,n-1}\frac{s_d}{\sqrt{n}}
$$

For the five lab pairs:

$$
\bar{d}=0.1133, \qquad s_d=0.01648
$$

$$
95\%\ \text{interval}=[0.0929,0.1338]
$$

This interval measures training-seed variation under the fixed generated-data procedure.

## Slide 19: Agreement criteria

| Target | Criterion |
| --- | --- |
| source and split files | exact SHA-256 |
| deterministic metric | absolute numerical tolerance |
| stochastic training | repeated-run or paired-effect agreement |
| hardware-sensitive timing | directional or behavioral agreement |

Choose the criterion before the reproduction outcome is known.

## Slide 20: Experiment manifest

```json
{
  "schema": "cme705-week13-v1",
  "files": {
    "config.json": "...sha256...",
    "runs.csv": "...sha256...",
    "summary.json": "...sha256..."
  },
  "numeric_tolerance": 1e-10
}
```

The manifest makes file integrity and interpretation testable.

## Slide 21: Hashes and scientific validity

A matching hash supports byte identity.

It does not answer whether:

- labels are correct;
- the split prevents leakage;
- the metric represents the intended construct;
- the comparison controls other factors;
- the claim generalizes.

Integrity is one layer of the evidence chain.

## Slide 22: Lab 13 workflow

```bash
python labs/week13_reproducibility_audit.py --audit-only
python labs/week13_reproducibility_audit.py
python labs/week13_reproducibility_audit.py \
  --verify outputs/week13/reference
```

The lab writes configuration, environment, run records, summary, and manifest files beneath ignored outputs.

## Slide 23: Threats to validity

| Threat | Main question | Project example |
| --- | --- | --- |
| construct | Does the metric represent the concept? | accuracy hides rare-class failure |
| internal | Could another factor explain the result? | preprocessing changed |
| statistical conclusion | Does the analysis support comparison? | one seed appears stable |
| external | Where should the result generalize? | one site supports a broad claim |

## Slide 24: A bounded limitation

Use this structure:

> Because [evidence boundary], the result does not establish [affected claim]. This may [likely consequence]. Test [specific next experiment].

Example:

> The test set contains one device type, so the result does not establish robustness to a new sensor. Evaluate a device-held-out split next.

## Slide 25: Peer reproduction protocol

Author:

- freezes a commit;
- supplies one command, target result, tolerance, and compute estimate;
- remains silent during the first attempt.

Reviewer:

- starts from a fresh environment;
- preserves commands and output;
- records the first material discrepancy;
- reruns after a committed correction.

## Slide 26: Reproduction status record

| Status | Meaning |
| --- | --- |
| reproduced | target agrees under the declared criterion |
| partial | some target outputs agree and scope is explicit |
| inconclusive | access or resources prevent the test |
| different | execution completes but result exceeds tolerance |
| blocked | setup cannot reach the target output |

Status describes evidence. It is not a grade by itself.

## Slide 27: Focused review comment

A useful comment states:

1. location;
2. observed evidence;
3. consequence for reproduction or claim;
4. smallest requested correction;
5. verification command.

A general complaint gives the author no testable next action.

## Slide 28: Draft report anatomy

The report aligns:

- research question and bounded contribution;
- related work and data provenance;
- baseline, method, and selection protocol;
- results, uncertainty, and error evidence;
- limitations and threats to validity;
- reproduction statement;
- next research experiment.

Every table and figure needs a source or generating command.

## Slide 29: From limitation to research direction

$$
\text{limitation}
+\text{candidate mechanism}
+\text{discriminating experiment}
+\text{decision rule}
$$

A credible master's or PhD direction requires:

- a meaningful unresolved gap;
- feasible data, skills, compute, and approvals;
- an experiment that can separate competing explanations.

## Slide 30: Week 13 deliverable and exit ticket

Submit:

- independent reproduction record;
- focused review comment;
- corrected commit and verification result;
- revised report draft;
- bounded next research experiment.

Exit ticket: What did reproduction establish, what remains uncertain, and which experiment could change the current conclusion?