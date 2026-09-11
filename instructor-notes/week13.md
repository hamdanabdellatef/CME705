# Instructor guide — Week 13

## Purpose

Week 13 converts the semester project into a research artifact that another student can execute and critique. The class should distinguish four outcomes:

- files are available;
- files are intact;
- a target result can be regenerated;
- the experimental design supports the claim.

The workshop succeeds when students preserve a failed first attempt, diagnose one material discrepancy, make a small correction, and verify it. Do not reward a polished result that hides undocumented repair.

## Preparation checklist

Before class:

1. Run `python labs/week13_reproducibility_audit.py --audit-only`.
2. Run the full lab and inspect `outputs/week13/reference/`.
3. Confirm students have assigned author-reviewer pairs.
4. Require authors to freeze a commit at least one day before the workshop.
5. Check that exchanges contain no credentials, personal identifiers, grades, restricted data, or private model access tokens.
6. Ask authors to estimate runtime, storage, and accelerator requirements.
7. Prepare a fallback synthetic bundle for students whose project data cannot be shared.
8. Decide how students will submit review records and draft reports.
9. Keep public course materials separate from student artifacts.

## Suggested timing

| Time | Activity | Instructor action |
| ---: | --- | --- |
| 0–10 min | Opening result audit | Ask what one displayed number depends on |
| 10–30 min | Claim-to-evidence chain | Trace claim, output, config, code, data, and environment |
| 30–50 min | Terminology cases | Require team, artifacts, and setup in every answer |
| 50–75 min | Artifact anatomy | Inspect the Week 13 manifest and result files |
| 75–100 min | Repeated-run calculation | Work paired differences and confidence interval |
| 100–120 min | Agreement criteria | Choose criteria before showing reproduction outcomes |
| 120–130 min | Break | Pair setup |
| 130–150 min | Lab 13 | Run audit, produce, and verify |
| 150–195 min | Peer first attempt | Authors remain silent except for access or safety problems |
| 195–215 min | Diagnosis and comment | Reviewers preserve evidence and request one change |
| 215–235 min | Revision and verification | Authors commit; reviewers rerun |
| 235–250 min | Report and validity audit | Connect one issue to a claim or limitation |
| 250–260 min | Research direction | Write one discriminating next experiment |

## Opening prompt

Display one number without context:

> Test accuracy: 0.81

Ask students to list what they need before interpreting it. Expected requests include:

- task and unit of observation;
- dataset version and test split;
- class distribution;
- metric averaging;
- baseline;
- selection procedure;
- repeated-run variation;
- code and configuration;
- compute and precision;
- comparison target;
- intended population.

The goal is to show that a number becomes evidence only within a protocol.

## Board sequence

### 1. Evidence graph

Write:

$$
C\leftarrow A\leftarrow O\leftarrow(K,\Theta,D,E).
$$

Ask one student to name each object. Then remove one edge and ask which claim becomes unsupported.

Use a project figure as a second example. Trace the figure to raw output and its generating command.

### 2. Terminology

Use the course definitions:

| Term | Team | Setup |
| --- | --- | --- |
| repeatability | same | same artifacts and setup |
| reproducibility | different | author artifacts and declared setup |
| replicability | different | independently created artifacts or setup |

Acknowledge that other fields use different labels. Grade the described actors and artifacts rather than the word alone when a student states a different convention explicitly.

### 3. Integrity

Write:

$$
H_i=\operatorname{SHA256}(\text{file}_i).
$$

Ask what a matching hash supports. Strong answer: byte identity under the named hash function. It does not establish label quality, split validity, metric choice, absence of leakage, or generalization.

### 4. Repeated-run calculation

Use the lab differences:

$$
[0.1111,0.1222,0.1111,0.1333,0.0889].
$$

Expected values:

| Quantity | Value |
| --- | ---: |
| $n$ | 5 |
| $\bar{d}$ | 0.1133 |
| $s_d$ | 0.01648 |
| standard error | 0.00737 |
| $t_{0.975,4}$ | 2.776 |
| margin | 0.02046 |
| interval | [0.0929, 0.1338] |

Interpretation: the interaction improves accuracy under the generated population, fixed split, implementation, and paired training seeds. The interval quantifies seed variation under that procedure. It excludes uncertainty across datasets, labels, domains, and collection processes.

### 5. Agreement criteria

Draw four rows:

- exact bytes;
- numerical tolerance;
- repeated-run agreement;
- directional or qualitative behavior.

Give examples before students choose criteria for their projects. Do not allow a criterion to be silently widened after the outcome is known.

### 6. Threats to validity

Use one question per category:

- construct: does the metric represent the intended concept?
- internal: could another factor explain the difference?
- statistical conclusion: does the analysis support the comparison?
- external: where should the conclusion generalize?

Require a consequence and next check. A list of generic threats is insufficient.

## Lab 13 facilitation

### Audit

Run:

```bash
python labs/week13_reproducibility_audit.py --audit-only
```

Expected markers:

```text
week13_reproducibility_audit
canonical_hash_stable=True
records=4 paired_runs=2
split_sizes=180/60/60
split_disjoint=True
files_written=False
downloads_started=False
```

The audit checks functions and partition logic without creating files.

### Produce and verify

Run:

```bash
python labs/week13_reproducibility_audit.py
```

Expected default evidence includes:

```text
files_hashed=4
mean_paired_accuracy_difference=0.1133
paired_95_percent_t_interval=0.0929,0.1338
verification_status=PASS
```

Values are deterministic for the declared NumPy path. Small timing differences are expected and deliberately excluded from hashed run records.

### Bundle inspection

Open:

- `config.json`;
- `environment.json`;
- `runs.csv`;
- `summary.json`;
- `manifest.json`.

Ask students to map every summary value back to seed-level records. Then map each record to the configuration and source hash.

### Tamper demonstration

Make a copy of the ignored output directory. Change one byte in `runs.csv`, then run:

```bash
python labs/week13_reproducibility_audit.py --verify path/to/copied-bundle
```

Expected outcome: verification stops with `Artifact hash mismatch: runs.csv`.

Explain that the check detects a changed file. It does not decide whether the original numbers were scientifically appropriate.

### Design audit

Ask students to name the controlled factor: inclusion of the $x_0x_1$ feature.

Fixed controls include data, split, seed pairs, optimizer, learning rate, batch size, epoch budget, selection rule, $L_2$ penalty, and metric implementation.

Ask why the best epoch can differ across seeds and conditions while the comparison remains controlled. The selection procedure is fixed, while its outcome responds to training.

## Peer workshop protocol

### Author responsibilities

The author shares a fixed commit and remains silent during the first attempt. They may intervene immediately for:

- accidental exposure of private information;
- unsafe commands;
- a credential request;
- an access rule that the reviewer could violate;
- a hardware condition that risks damage.

These interventions should be recorded.

### Reviewer responsibilities

The reviewer:

1. records the starting state;
2. creates a fresh environment;
3. follows the written command;
4. preserves output and errors;
5. compares the target under the declared criterion;
6. writes one focused comment;
7. reruns after the author commits a fix.

The reviewer should avoid broad redesign requests during the reproduction check. The target is one central result.

### Status labels

- **reproduced:** target agrees;
- **partial:** some target outputs agree and scope is explicit;
- **inconclusive:** access or resources prevent the test;
- **different:** procedure completes but exceeds tolerance;
- **blocked:** procedure cannot reach the target output.

A blocked result can reveal a high-value documentation problem.

## Review-comment examples

### Too vague

> The code does not work.

### Evidence-based

> At commit `abc123`, `python train.py` exits because `data/split.json` is absent and the README provides no creation command. This blocks the declared test result. Add the split-generation command or permitted download route and its expected hash. I will recreate the environment and rerun the training command.

### Unsupported diagnosis

> The result changed because PyTorch is nondeterministic.

### Bounded diagnosis

> The author used PyTorch 2.6 and CUDA 12.6, while the requirements allow any PyTorch 2.x version. The output differs by 0.004, above the stated $10^{-4}$ tolerance. Pin the final-run version or widen the tolerance with repeated-run evidence. I will test the pinned version first.

## Worksheet answer guide

### Terminology

- Case A: repeatability.
- Case B: reproducibility.
- Case C: replicability under the course convention.

Accept another convention when the student explicitly describes the team, artifacts, and setup.

### Integrity

Matching hashes support byte identity. Missing scientific checks include data validity, leakage, metric validity, controlled comparison, uncertainty, and generalization.

### Paired calculation

Expected interval: [0.0929, 0.1338]. Allow rounding differences at the fourth decimal.

### Tamper prediction

The verifier should stop before rerunning the experiment because the recorded file hash differs.

## Draft-report workshop

Pair students again after reproduction. Ask them to compare four locations:

1. title and abstract;
2. results table;
3. limitations;
4. conclusion.

Highlight claims that appear in only one location or expand in scope between results and conclusion.

For one table or figure, require:

- source or generating command;
- data split;
- metric and units;
- uncertainty definition;
- configuration or result ID;
- caption explaining the comparison.

## Assessment

Use the expanded Assignment 5 rubric. Prioritize:

- faithful first attempt;
- preserved evidence;
- diagnosis that separates plausible causes;
- a focused comment;
- correction at a new commit;
- verification after correction;
- report claims revised to match evidence.

Do not grade reproduction status as success versus failure. Grade the quality of the empirical review.

## Common misconceptions

### “A seed guarantees identical results”

A seed initializes random generators. Library versions, algorithms, parallel ordering, and nondeterministic kernels can still change results.

### “A lock file reproduces the experiment”

A lock file records software resolution. The experiment also depends on data, splits, configuration, code, external services, hardware, and commands.

### “Matching hashes prove the result”

Hashes detect byte changes. They do not validate the data, design, metric, or inference.

### “A different result means the paper is wrong”

The difference may arise from incomplete instructions, hidden state, environment drift, stochastic variation, hardware, or a real scientific boundary. Diagnose before concluding.

### “More seeds solve generalization”

More training seeds estimate algorithmic variation under the fixed data procedure. They do not substitute for new sites, time periods, devices, or populations.

### “Peer review should redesign the project”

The Week 13 reproduction review targets one central result and one material correction. Broader suggestions belong in a separate discussion.

## Exit-ticket answer guide

A strong response says that successful reproduction establishes that a specified result can be regenerated from the tested artifacts and environment within a declared criterion. It names the immutable commit, data or split identity, and environment. It distinguishes integrity and computational agreement from validity and generalization. It turns a specific limitation into an experiment with a metric and result that could change the current conclusion.