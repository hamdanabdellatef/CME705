# Week 13 notes — Reproducible Research, Uncertainty, and Peer Review

## 1. A result is a traceable claim

A machine-learning report usually contains several kinds of statements. They require different evidence:

- **observation:** a recorded quantity from one run;
- **summary:** a calculation across observations;
- **comparison:** a contrast under declared controls;
- **inference:** an explanation supported by the comparison;
- **scope claim:** a statement about populations, domains, or future cases.

Consider:

> The interaction model achieved 0.633 test accuracy for seed 13.

This is an observation. It becomes traceable when the report names the code revision, data split, configuration, command, output file, metric definition, and test role.

Now consider:

> Adding the interaction improves accuracy.

This is a comparative claim. It needs a baseline, a controlled treatment, repeated measurements, a fixed evaluation protocol, and a bounded population. One run cannot distinguish the proposed mechanism from seed variation.

### 1.1 Claim scope

Claims can become broader along several axes:

| Axis | Narrow statement | Broader statement requiring more evidence |
| --- | --- | --- |
| data | this fixed test set | the target population |
| time | this collection period | future conditions |
| location | this institution | other institutions |
| hardware | this GPU and software stack | other platforms |
| randomness | this seed | typical training behavior |
| model | this checkpoint | the architecture family |
| metric | accuracy | usefulness or safety |

A precise report states the narrowest claim that the evidence supports. It may then motivate a broader hypothesis for future work.

## 2. The evidence graph

The trace from a claim to its source can be represented as a directed graph:

$$
C
\leftarrow A
\leftarrow O
\leftarrow (K,\Theta,D,E),
$$

where:

- $C$ is the claim;
- $A$ is the analysis;
- $O$ is the recorded output;
- $K$ is code at a fixed version;
- $\Theta$ is the configuration;
- $D$ is data and split identity;
- $E$ is the execution environment.

A result table should permit a reader to follow each edge. The report may use a compact result ID such as `table2-row3`, then record the generating command and output path in an appendix or manifest.

### 2.1 Provenance questions

For any displayed number, ask:

1. Which data split produced it?
2. Which metric implementation and version calculated it?
3. Which checkpoint or stopping rule selected the model?
4. Which command generated the result file?
5. Which code revision and configuration were active?
6. Which transformation converted raw output into the displayed value?
7. What variation or uncertainty does the number omit?

If the answer depends on memory or a private notebook cell, the provenance chain is incomplete.

## 3. Repeatability, reproducibility, and replicability

Different communities reverse some labels. This course follows the ACM artifact-review framing and records the underlying situation so the meaning remains clear.

### 3.1 Repeatability

The same team uses the same artifacts and setup. Examples include rerunning a saved command or repeating training with the same environment and seeds.

Repeatability can expose hidden state, missing randomness controls, nondeterministic operations, or overwritten inputs.

### 3.2 Reproducibility

A different team uses the author-provided artifacts and declared setup. In Week 13, one student checks out a peer's fixed commit and follows only the written instructions during the first attempt.

Reproducibility tests the artifact package and instructions. It also provides an opportunity to inspect whether the outputs actually support the stated claim.

### 3.3 Replicability

A different team independently constructs artifacts or an experimental setup to test the same claim. A replication may use a reimplementation, new dataset, different site, or new measurement procedure.

Replication tests a broader part of the scientific claim. Agreement can strengthen confidence. Disagreement can reveal boundary conditions or competing explanations.

### 3.4 Report the situation

Always record:

- original and reviewing teams;
- whether author code, data, weights, and scripts were used;
- which environment and hardware changed;
- whether the target was an exact number, interval, ordering, or qualitative behavior.

## 4. What a successful rerun establishes

A successful computational rerun supports statements such as:

- the declared command executes in the tested environment;
- the provided artifacts are internally consistent;
- the result can be regenerated within the declared tolerance;
- the report points to the correct output.

Further questions remain:

- Were the labels and data collection appropriate?
- Was test data protected from selection?
- Does the metric measure the intended construct?
- Was the comparison controlled?
- Does the uncertainty method match the sampling process?
- Does the claim extend beyond the tested conditions?
- Were licensing, consent, privacy, and release constraints respected?

Computational reproducibility is necessary for many empirical claims, while research validity requires additional evidence.

## 5. Repository identity

A branch name such as `main` moves. A commit identifier names a specific repository state.

Record:

```text
repository: https://example.org/research/project
commit: 4b11d2...
entry command: python train.py --config configs/final.json
expected output: outputs/final/metrics.json
```

A tagged release can improve discoverability, but the tag should resolve to a documented commit. An archive should include a cryptographic checksum.

### 5.1 Dirty working trees

Uncommitted changes break the link between a commit and the executed code. Before a final run, record:

```bash
git status --short
git rev-parse HEAD
```

If a dirty tree is necessary, save a patch or archive checksum and explain why. The preferred workflow commits the exact code and configuration before generating the final result.

## 6. Environment identity

A reproducible environment needs both a construction recipe and an observation of what actually ran.

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

Record operating system, Python version, major library versions, hardware type, accelerator model, driver or runtime when relevant, memory constraints, and elapsed time.

A permissive requirement such as `torch>=2` supports installation but may not reproduce an old result. A lock file or exported environment captures resolved versions. Do not assume a lock file makes code portable across all operating systems and hardware.

### 6.1 Containers

A container can preserve user-space dependencies and commands. It does not automatically preserve:

- external data;
- GPU drivers and hardware behavior;
- network services;
- credentials;
- licensing permission;
- random or nondeterministic kernels;
- long-term availability of base images.

Document these boundaries.

## 7. Data and split identity

A dataset name rarely identifies exact bytes. Record:

- source and access route;
- version, date, or immutable release ID;
- license and use constraints;
- raw-file checksums when redistribution is permitted;
- inclusion and exclusion rules;
- unit of observation;
- label construction;
- preprocessing fitted on training data;
- train, validation, and test identifiers or split-generation procedure;
- any group, temporal, or geographic separation.

For private data, publish metadata, schema, split code, synthetic tests, and access instructions when allowed. Never place private records in a public artifact merely to simplify reproduction.

### 7.1 Split checksum

If a split file contains ordered identifiers $I=(i_1,\ldots,i_m)$, save a canonical representation and compute

$$
H_{\mathrm{split}}=\operatorname{SHA256}(I).
$$

The hash detects a byte-level change. A reviewer must still check that the split design prevents leakage and matches the intended evaluation population.

## 8. Configuration identity

A source file shows possible settings. A result requires the settings actually used.

Capture:

- model and representation;
- initialization and seed;
- loss and regularization;
- optimizer and schedule;
- batch size and epoch budget;
- augmentation;
- selection metric and checkpoint rule;
- stopping rule;
- inference and threshold settings;
- hardware and precision;
- evaluation implementation;
- all deviations from the plan.

Prefer a machine-readable configuration saved beside the output. Print the resolved configuration at run start, after defaults and command-line overrides have been applied.

## 9. Randomness and determinism

Random seeds are part of the experiment, not a guarantee of identical output.

Sources of variation include:

- initialization;
- sample ordering;
- data augmentation;
- stochastic layers;
- parallel reduction order;
- nondeterministic accelerator kernels;
- asynchronous input pipelines;
- version-specific algorithms.

Use a seed set chosen before comparing conditions. Record deterministic settings when used. If exact determinism is unavailable or too costly, declare the expected tolerance and report repeated-run variability.

### 9.1 Seed pairing

Suppose condition A and condition B share seed $i$. Their paired difference is

$$
d_i=x_i^{(B)}-x_i^{(A)}.
$$

Pairing removes variation that both conditions share. It is effective when the pairing corresponds to the same data split, initialization logic, or other meaningful random component.

Using different seed sets for two conditions weakens the comparison and can confound treatment with random variation.

## 10. Mean and sample standard deviation

For results $x_1,\ldots,x_n$,

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i.
$$

The sample variance and standard deviation are

$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2,
$$

$$
s=\sqrt{s^2}.
$$

The denominator $n-1$ estimates population variation from a sample. State whether reported variation is standard deviation, standard error, confidence interval, or another quantity.

### 10.1 Standard error

The estimated standard error of the mean is

$$
\operatorname{SE}(\bar{x})=\frac{s}{\sqrt{n}}.
$$

Standard deviation describes run-to-run spread. Standard error describes uncertainty in the estimated mean under the sampling assumptions. They answer different questions.

## 11. Paired confidence interval

Let $\bar{d}$ and $s_d$ denote the mean and sample standard deviation of $n$ paired differences. A two-sided 95% Student-$t$ interval is

$$
\bar{d}
\pm
t_{0.975,n-1}\frac{s_d}{\sqrt{n}}.
$$

The interpretation depends on how runs were sampled. The interval does not describe uncertainty from new datasets or domains unless that variation entered the sampling procedure.

### 11.1 Week 13 worked example

The lab records paired test-accuracy differences:

$$
[0.1111,\ 0.1222,\ 0.1111,\ 0.1333,\ 0.0889].
$$

Their mean is

$$
\bar{d}=0.1133,
$$

and their sample standard deviation is

$$
s_d=0.01648.
$$

For $n=5$, $t_{0.975,4}=2.776$. The standard error is

$$
\frac{0.01648}{\sqrt{5}}\approx0.00737.
$$

The margin is

$$
2.776(0.00737)\approx0.02046.
$$

Therefore,

$$
0.1133\pm0.02046=[0.0929,0.1338].
$$

The interval supports a positive effect under this generated dataset, fixed split, implementation, and seed-pairing procedure. It does not establish the effect on a real dataset.

## 12. Choosing an agreement criterion

Reproduction targets differ.

### 12.1 Exact agreement

Use byte equality or hashes for immutable source, configuration, split files, and static assets.

$$
H(x)=H(x')
$$

supports byte equality when a secure cryptographic hash is used correctly.

### 12.2 Numerical agreement

For a metric $m$, declare an absolute or relative tolerance:

$$
|m'-m|\le\epsilon.
$$

Choose $\epsilon$ from numeric precision, expected nondeterminism, metric resolution, and the scientific decision boundary. Do not choose it after seeing the discrepancy.

### 12.3 Behavioral agreement

Some claims concern direction or ordering. A systems study may expect method B to remain faster, even when absolute seconds change with hardware. Define which comparisons must hold and what change would count as disagreement.

### 12.4 Statistical agreement

For stochastic experiments, compare the repeated-run distribution or paired effect. Overlapping confidence intervals alone do not constitute a general hypothesis test. Report the method and assumptions used.

## 13. Artifact manifests

The Week 13 lab writes a manifest resembling:

```json
{
  "schema": "cme705-week13-v1",
  "files": {
    "config.json": "...sha256...",
    "environment.json": "...sha256...",
    "runs.csv": "...sha256...",
    "summary.json": "...sha256..."
  },
  "source": {
    "path": "labs/week13_reproducibility_audit.py",
    "sha256": "...sha256..."
  },
  "numeric_tolerance": 1e-10
}
```

The schema says how to interpret the package. File hashes detect changes. The source hash connects the output to the implementation. The tolerance defines numerical agreement before verification.

A manifest should not contain secrets, access tokens, private paths, or personal data.

## 14. The Week 13 lab

The lab generates one fixed nonlinear binary-classification population. It compares:

- baseline: three linear features;
- treatment: the same features plus one prespecified interaction $x_0x_1$.

The controlled factor is the interaction feature. Both conditions share:

- generated observations and labels;
- stratified train, validation, and locked test partitions;
- training-seed set;
- optimizer, learning rate, batch size, epoch budget, and $L_2$ penalty;
- validation checkpoint selection;
- metric implementation.

Run:

```bash
python labs/week13_reproducibility_audit.py
```

The command first creates the bundle and then verifies every hash, the source, split identity, and numerical summary.

Independent verification uses:

```bash
python labs/week13_reproducibility_audit.py --verify outputs/week13/reference
```

### 14.1 Interpreting verification

A `PASS` result means the files are intact and the experiment regenerated the selected numerical summary within tolerance in the tested environment.

It does not prove that:

- the generated task represents an application;
- accuracy is the best metric;
- the statistical assumptions are universally valid;
- the interaction is novel;
- the result generalizes beyond the fixed data-generating process.

## 15. Threats to validity

### 15.1 Construct validity

Does the operational measurement represent the intended concept?

Examples:

- accuracy hides rare-class recall;
- a proxy label differs from the real outcome;
- FID does not measure factuality or memorization;
- a questionnaire score does not capture the full construct.

### 15.2 Internal validity

Could another factor explain the observed difference?

Examples:

- preprocessing differs between conditions;
- test data influence model selection;
- compute budgets differ;
- augmentation changes with architecture;
- one implementation contains a bug.

### 15.3 Statistical conclusion validity

Does the analysis support the stated comparison?

Examples:

- one seed is treated as stable evidence;
- unpaired runs are analyzed as paired;
- repeated test evaluation invalidates independence;
- multiple comparisons are ignored;
- variation source is unnamed.

### 15.4 External validity

Where should the conclusion generalize?

Examples:

- one hospital does not establish cross-hospital behavior;
- one language does not support a multilingual claim;
- one benchmark may not represent deployment data;
- a short sequence result may not extend to long contexts.

### 15.5 Limitation structure

Write:

> Because [design or evidence boundary], the result does not establish [affected claim]. This may [likely consequence]. Test [specific next experiment].

This structure links the limitation to a decision.

## 16. Peer reproduction as a protocol

### 16.1 Author preparation

The author freezes a commit, checks the public artifact for private content, and provides:

- setup command;
- run command;
- expected outputs;
- expected result and tolerance;
- compute estimate;
- data access instructions;
- known limitations.

### 16.2 Reviewer first attempt

The reviewer starts from a fresh environment and uses only written instructions. They preserve command output and record the first point where the procedure becomes ambiguous or fails.

The reviewer does not silently repair the artifact. A hidden repair prevents the author from learning what another reader will encounter.

### 16.3 Diagnosis

Classify the result:

| Status | Meaning |
| --- | --- |
| reproduced | target result agrees under the declared criterion |
| partial | some target outputs agree; scope is explicit |
| inconclusive | resource or access limit prevents the test |
| different | command runs but result exceeds tolerance |
| blocked | setup or execution cannot reach the target output |

A status is descriptive, not a grade by itself.

### 16.4 Revision and verification

The author makes the smallest correction that addresses the observed problem. The reviewer reruns the relevant step and records the new commit. A resolved comment includes the verification evidence.

## 17. Writing review comments

A focused comment contains:

1. location;
2. observed evidence;
3. consequence;
4. requested change;
5. verification plan.

Weak:

> The README is unclear.

Strong:

> The command under “Training” omits `--config configs/final.json`. At commit `abc123`, the default uses a random row split and produces a different test set. Add the exact config path and expected split hash. I will rerun the command and compare `metrics.json` within the declared tolerance.

The strong comment avoids guessing intent and connects the change to a reproducible test.

## 18. Draft-report structure

A concise project report should contain:

1. title and abstract;
2. problem, population, task, and intended use;
3. research question and bounded contribution;
4. related-work comparison;
5. data provenance and split protocol;
6. baseline and method;
7. experimental and selection protocol;
8. results with uncertainty and error evidence;
9. limitations and threats to validity;
10. conclusion and research direction;
11. reproducibility statement;
12. references and optional appendices.

Every result should say whether it is published prior work or produced in the project.

### 18.1 Abstract audit

The abstract should answer:

- What problem was studied?
- Which data and method were used?
- What is the main result, with metric and scope?
- What conclusion follows?
- Which limitation materially bounds that conclusion?

Do not add novelty, generality, or deployment claims that the body does not establish.

## 19. Tables and figures

A result table needs:

- dataset and split;
- metric and direction;
- conditions and controls;
- repeated-run summary;
- uncertainty definition;
- unit and precision;
- result source or generating command.

A figure caption should state what the reader sees, how it was produced, and which comparison it supports. Axes need labels and units. Error bars need a definition.

## 20. From limitation to research direction

A useful next question combines:

$$
\text{limitation}
+\text{mechanism}
+\text{experiment}
+\text{decision rule}.
$$

Example:

> Performance falls on data from a new sensor. We hypothesize that the frequency response shifts the learned features. Compare calibration, domain-adversarial training, and an unchanged baseline on a device-held-out split. Continue with domain adaptation only if it improves macro recall across all held-out devices without increasing calibration error beyond the declared threshold.

This could develop into master's or PhD work if the gap is novel, the data and approvals are feasible, the mechanism is scientifically meaningful, and the planned experiments can distinguish competing explanations.

## 21. Responsible artifact release

Before public release:

- remove credentials and private paths;
- review data, code, model, and media licenses;
- document restricted assets without redistributing them;
- disclose consent or institutional review where applicable;
- avoid publishing identifying error examples;
- record automated tools when they affect the method or scientific result;
- explain model misuse or downstream risks when relevant;
- provide contact and issue-reporting routes.

A reproducible procedure does not override privacy, consent, copyright, or security obligations.

## 22. Claims supported by Week 13

After the lesson and lab, students can support these statements:

- a result can be traced from claim to data, configuration, code, and environment;
- a different researcher can test an author-provided artifact under a declared criterion;
- repeated paired runs quantify one named source of variation;
- hashes detect byte changes but do not establish scientific validity;
- a review comment can locate a problem, explain its consequence, and request a verifiable fix;
- a limitation can motivate a discriminating next experiment.

The workshop does not establish that every project is valid, generalizable, ethical, or ready for deployment. Those conclusions require project-specific evidence.