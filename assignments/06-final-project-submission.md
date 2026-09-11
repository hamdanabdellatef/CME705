# Assignment 6: Final Project Release, Presentation, and Research Direction

Submit a fixed, reviewable project release and defend one central contribution in an 8-minute presentation. The final package should let another researcher trace the evidence, reproduce one result, understand the limits, and evaluate the proposed next study.

## Deliverables

Submit:

1. repository URL and exact final commit or permitted archive with SHA-256;
2. final 5–7 page report and editable source;
3. 8–10 main presentation slides, optional backup slides, and PDF export;
4. completed [final-release checklist](../research-project/final-release-checklist.md);
5. completed [research-direction note](../research-project/research-direction-note.md);
6. Week 13 reproduction record, correction, and verification status;
7. `project-release.json` based on [the example manifest](../research-project/project-release.example.json);
8. output from the Week 14 static release audit; and
9. a concise self-assessment against the [project rubric](../research-project/rubric.md) and [presentation rubric](../research-project/presentation-rubric.md).

Use the institutional submission route and deadline announced by the instructor.

## Final repository release

The repository or permitted archive must contain:

- a landing README with the research question and supported claim;
- environment construction and supported Python version;
- data source, version, access, license, population, and split;
- final configuration and seed set;
- one command that regenerates one central result;
- expected output path and agreement criterion;
- report and presentation files;
- code and tests needed for the central result;
- license and citation information;
- known limitations and inappropriate uses;
- research-direction note; and
- no credentials, private records, generated caches, or prohibited data.

A private repository or restricted package is acceptable when access conditions require it. Explain the access route and what can be reviewed.

## Release freeze

Before recording the final identifier:

```bash
git status --short
python -m pytest
python labs/week14_release_audit.py --project-root . --manifest project-release.json
git rev-parse HEAD
```

Run the project-specific reproduction command from a fresh environment or preserve the Week 13 independent verification. If the release audit or reproduction fails, record the failure and resolution before freezing the final commit.

## Report standard

Use the [report template](../research-project/report-template.md). The final report must include:

- bounded question and contribution;
- related-work comparison;
- data provenance and access;
- baseline and method;
- leakage-safe selection and evaluation;
- controlled comparison, ablation, or robustness check;
- uncertainty and error evidence;
- limitations tied to affected claims;
- result provenance;
- reproduction statement; and
- next discriminating experiment.

The title, abstract, main result, conclusion, and presentation must agree in scope.

## Presentation standard

Recommended format:

- 8 minutes presentation;
- 4 minutes questions;
- 8–10 main slides;
- backup slides as needed.

The talk should answer:

1. What matters?
2. What was tested?
3. How was it tested?
4. What happened?
5. What does the result establish?
6. What remains unresolved?
7. What should be tested next?
8. How can one result be reproduced?

Use [the student presentation template](../research-project/presentation-template.md) as a starting point. The template is guidance rather than a required visual style.

## Central evidence slide

The main result slide must identify:

- evaluated population and split;
- metric and direction;
- compared conditions;
- sample, fold, site, or seed count;
- uncertainty or individual observations;
- whether the result is validation, locked test, or exploratory;
- project-generated or published provenance; and
- generating command or result identifier.

A section title such as “Results” is insufficient. State the conclusion the audience should inspect.

## Questions and defense

Answer with:

1. observation;
2. interpretation;
3. uncertainty or limitation; and
4. next evidence.

Correct an overbroad premise before answering it. When an answer is unknown, state what evidence would resolve it.

## Research-direction note

The note must begin with an observed limitation or unexplained pattern. Include:

- two plausible explanations;
- one minimum experiment that produces different predictions;
- fixed controls;
- metric and uncertainty unit;
- result that favors each explanation;
- inconclusive region or stop condition;
- data, skills, compute, collaborators, and approvals;
- 90-day first stage; and
- current classification as feasibility, master's, PhD, or applied research.

Institutional requirements and supervisor judgment determine eventual degree scope.

## Static release audit

The audit checks the declared package without executing the reproduction command:

```bash
python labs/week14_release_audit.py --audit-only
python labs/week14_release_audit.py --project-root path/to/project --manifest project-release.json
```

Add `--verify-git` only when the manifest and project root identify the intended Git repository.

A passing audit establishes internal consistency of the declared files and hashes. It does not establish scientific validity or replace independent reproduction.

## Self-assessment

For each rubric criterion, record:

- claimed level;
- file, slide, result, or answer supporting the claim;
- one remaining weakness.

Do not award yourself points only for model complexity or amount of code.

## Assessment

The final-project score uses [the project rubric](../research-project/rubric.md). The oral presentation uses [the presentation rubric](../research-project/presentation-rubric.md). The instructor may announce their relative contribution to the course project grade before the offering.

Essential conditions:

- test data were not used silently for selection;
- external and project-generated results are distinct;
- every central result has provenance;
- private or restricted data are handled according to policy;
- the student can explain and defend the submitted work;
- the final identifier resolves to the assessed artifact.

## Submission record

| Item | Value |
| --- | --- |
| repository or archive | |
| final commit or SHA-256 | |
| release tag, if used | |
| report file | |
| presentation file | |
| reproduction command | |
| expected output | |
| agreement criterion | |
| Week 13 verification | |
| Week 14 audit | |
| data access status | |
| research-direction file | |

## Final check

- [ ] the working state is frozen and identified;
- [ ] report and presentation open correctly;
- [ ] one central result can be traced and reproduced;
- [ ] data and model licenses are recorded;
- [ ] generated or assisted material is attributed under the course policy;
- [ ] limitations change the scope of the conclusion where necessary;
- [ ] the next experiment distinguishes competing explanations;
- [ ] no secrets, private identifiers, or prohibited files are included;
- [ ] the correct institutional submission route was used.
