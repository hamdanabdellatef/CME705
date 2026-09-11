# Assignment 5: Reproducibility Review and Draft-Report Revision

A classmate will test whether one central result can be regenerated from your fixed project artifact. You will review a peer's artifact, write one focused comment, and verify the correction. You will then revise your own report so its claims, evidence, limitations, and reproduction instructions agree.

## Deliverables

Submit:

1. a completed peer-reproduction record of 1,000–1,500 words, excluding command output and tables;
2. one focused issue or review comment containing location, evidence, consequence, requested change, and verification plan;
3. the reviewed author's corrected commit and the reviewer's verification result;
4. your revised 5–7 page project-report draft using the [report template](../research-project/report-template.md);
5. a short author response listing changes made and unresolved limitations.

A failed or partial reproduction can earn full credit when the attempt is faithful, the evidence is preserved, the diagnosis is bounded, and the correction is tested.

## Roles

Each student acts once as author and once as reviewer.

### Author package

Freeze a commit before the exchange. Supply:

- repository URL and exact commit;
- supported operating system and Python version;
- one environment-creation command;
- one command that reproduces one central table row or figure;
- exact configuration and seed set;
- dataset source, version, access rules, and split identity;
- expected output paths;
- expected metric or behavior and a prespecified tolerance;
- estimated runtime, memory, storage, and accelerator needs;
- known nondeterminism and limitations.

Remove credentials, personal identifiers, private records, licensed files that cannot be redistributed, and machine-specific secrets. For restricted data, provide a permitted access route or a synthetic audit path rather than copying the data.

### Reviewer protocol

Start from a fresh environment. During the first attempt, follow the written instructions without verbal repair from the author.

Record:

- repository and commit actually reviewed;
- environment construction and resolved versions;
- commands and exit status;
- expected and observed files;
- expected result and agreement criterion;
- observed result, elapsed time, and compute device;
- every deviation from the supplied instructions;
- the first blocking or material ambiguity;
- whether the target was reproduced, partial, inconclusive, different, or blocked.

Preserve the original output before making a diagnosis.

## Agreement criteria

Declare the criterion before running:

- exact SHA-256 for immutable files;
- absolute or relative numerical tolerance for deterministic values;
- repeated-run or paired-effect agreement for stochastic training;
- directional or ordering agreement for hardware-sensitive systems.

Explain why the chosen tolerance matches the metric, numeric precision, hardware, and scientific decision. Do not widen it after seeing the discrepancy without labeling the change.

## Focused review comment

Use this structure:

**Location:** Identify the file, section, command, or result.

**Observation:** Report what happened with the commit, command, and relevant output.

**Consequence:** Explain which reproduction step or research claim is affected.

**Requested change:** Ask for the smallest correction that addresses the evidence.

**Verification:** State the command or comparison you will rerun.

Example:

> In `README.md`, the training command omits `--config configs/final.json`. At commit `abc123`, the default command constructs a different split and produces 0.71 rather than the expected 0.78. Please add the exact configuration path and expected split hash. I will rerun the corrected command and compare `metrics.json` under the declared tolerance.

## Diagnosis categories

Use evidence to distinguish:

- missing or ambiguous instructions;
- dependency drift;
- unavailable data, weights, or permissions;
- data-version or split mismatch;
- configuration mismatch;
- hidden state or cached artifacts;
- uncontrolled randomness;
- hardware or precision sensitivity;
- metric or analysis mismatch;
- implementation error;
- insufficient compute or storage.

When two explanations remain plausible, propose the smallest test that separates them.

## Revision cycle

1. The author responds to the evidence and makes a minimal correction.
2. The author commits the correction and shares the new identifier.
3. The reviewer reruns the affected step.
4. The reviewer records the new result and status.
5. Both students list limitations that remain unresolved.

Do not mark an issue resolved until the verification step passes or the remaining disagreement is documented.

## Draft-report revision

Use the peer review to audit your own report. The draft must include:

- a bounded research question and contribution;
- structured comparison with at least five credible sources;
- data provenance, license or access conditions, population, and split protocol;
- a simple baseline and justified method;
- training, selection, and locked-evaluation procedures;
- at least one controlled comparison, ablation, or robustness check;
- repeated-run or other appropriate uncertainty;
- error or subgroup evidence;
- limitations tied to affected claims;
- the generating command and provenance for each table or figure;
- a reproduction statement with commit, environment, data, commands, outputs, and compute;
- one next experiment that could change the current conclusion.

Published results must remain visually and verbally separate from your own results.

## Reproduction record template

### Identity

| Item | Value |
| --- | --- |
| author | |
| reviewer | |
| repository | |
| reviewed commit | |
| review date | |
| target result | |

### First attempt

| Item | Expected | Observed |
| --- | --- | --- |
| environment | | |
| command | | |
| output files | | |
| metric or behavior | | |
| runtime and device | | |

Agreement criterion: __________________________________________________________

Status: reproduced / partial / inconclusive / different / blocked

Evidence:

> ____________________________________________________________________________

### Material issue

- Location: __________________________________________________________________
- Observation: ________________________________________________________________
- Consequence: ________________________________________________________________
- Requested change: ___________________________________________________________
- Verification plan: __________________________________________________________

### Verification after correction

| Item | Value |
| --- | --- |
| corrected commit | |
| command rerun | |
| observed result | |
| final status | |
| unresolved limitation | |

## Assessment guide

| Criterion | Weight | Strong evidence |
| --- | ---: | --- |
| faithful independent attempt | 25% | Fresh setup, exact commit and commands, preserved output, no silent repair |
| diagnostic precision | 20% | Discrepancy is located and competing causes are separated by evidence |
| review comment | 15% | Specific consequence, minimal requested change, and executable verification |
| correction and verification | 15% | New commit is identified and the relevant step is rerun |
| draft-report evidence | 15% | Claims, tables, uncertainty, limitations, and provenance agree |
| communication and research direction | 10% | Respectful record and a feasible next experiment tied to an observed limitation |

## Submission check

- [ ] The reviewed commit is immutable and the working tree state is recorded.
- [ ] The first attempt used only written instructions.
- [ ] Commands, outputs, versions, device, and elapsed time are preserved.
- [ ] The agreement criterion was chosen before the result was observed.
- [ ] The review comment cites concrete evidence and requests a verifiable change.
- [ ] The correction has a new commit identifier.
- [ ] The reviewer reran the affected step.
- [ ] The report distinguishes published and project-generated results.
- [ ] Every central table or figure has a generating command or source.
- [ ] Limitations state the affected claim and likely consequence.
- [ ] The next experiment could change the current conclusion.
- [ ] No credentials, personal data, or restricted artifacts are included.