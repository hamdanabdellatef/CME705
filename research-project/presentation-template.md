# CME705 project presentation template

Use this storyboard for an 8-minute final presentation. Replace every bracketed prompt. Keep 8–10 main slides and place audit details in backup slides.

A PowerPoint version is available at [`slides/cme705-project-presentation-template.pptx`](slides/cme705-project-presentation-template.pptx).

## Slide 1 — Ask the research question

**Title:** [Question the project answers]

[Student name]<br>
CME705 Machine Learning<br>
[Repository release or short commit]

Say in one sentence why the question matters.

## Slide 2 — Define the problem and gap

**Evidence-bearing title:** [Population or decision faces a specific unresolved problem]

Show the intended use, unit of observation, target, affected population, and one gap supported by a primary source. Avoid a general history of the field.

## Slide 3 — State what was tested

**Evidence-bearing title:** [The project tests whether X changes Y under Z]

Include the bounded question, directional prediction, baseline, controlled factor, and contribution if the prediction is unsupported.

## Slide 4 — Identify the evidence protocol

**Evidence-bearing title:** [The evaluation isolates the intended comparison]

| Protocol item | Value |
| --- | --- |
| data version and access | |
| train / validation / test | |
| baseline | |
| selected metric | |
| model-selection rule | |
| seeds or uncertainty unit | |
| compute budget | |

## Slide 5 — Explain the necessary mechanism

**Evidence-bearing title:** [Mechanism that connects the method to the hypothesis]

Use one diagram. Label input, transformation, controlled change, and output. Move complete architecture and hyperparameters to backup.

## Slide 6 — Show the central result

**Evidence-bearing title:** [Observed comparison with direction and boundary]

Show conditions, metric and direction, sample or seed count, uncertainty or individual observations, validation or locked-test status, and source or result ID.

## Slide 7 — Qualify with errors or robustness

**Evidence-bearing title:** [Specific failure pattern, subgroup result, or stress-test finding]

Show one result that changes interpretation of Slide 6. State how examples were selected.

## Slide 8 — State conclusion and limitation

**Evidence-bearing title:** [Strongest supported claim and its boundary]

> Because [boundary], the project does not establish [affected claim]. This may [consequence].

Do not introduce a new result.

## Slide 9 — Propose a discriminating next experiment

**Evidence-bearing title:** [Experiment that separates explanation A from explanation B]

Show the observed boundary, both explanations and predictions, controlled experiment, metric, decision rule, resources, approvals, and current research stage.

## Slide 10 — Reproduce the evidence

**Title:** [Fixed release and central reproduction route]

```text
repository:
commit or release:
install:
command:
expected output:
agreement criterion:
runtime and device:
```

End with the strongest established claim, unresolved question, and contact or citation.

## Backup slides

Prepare only useful backup evidence:

1. split construction and counts;
2. architecture and parameter count;
3. full configuration;
4. learning curves and selection history;
5. full confusion matrix or residual analysis;
6. seed-level results and intervals;
7. additional robustness checks;
8. compute and environment;
9. data, model, and code licenses;
10. Week 13 reproduction record;
11. release manifest and hashes.

Label exploratory evidence.

## Speaker-note prompts

For each main slide, record target time, transition, sentence to remember, source or generating command, claim not established, likely question, and relevant backup slide.

## Final template check

- [ ] each main slide advances one argument;
- [ ] evidence slides use conclusion-bearing titles;
- [ ] central labels are readable when projected;
- [ ] citations and commands are preserved;
- [ ] color is not the only distinction;
- [ ] the talk finishes within eight minutes;
- [ ] PDF fallback opens correctly;
- [ ] no private or identifying material appears.
