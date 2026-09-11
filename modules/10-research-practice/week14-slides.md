# Week 14 slides — Research Presentation, Final Release, and Next Direction

Accessible source for the Week 14 instructor PowerPoint. Each section corresponds to one slide.

## Slide 1: Research presentation, final release, and next direction

CME705 Machine Learning<br>
Neural Networks, Deep Learning, and Research Practice<br>
Week 14

## Slide 2: Capstone evidence

- Present one supported contribution.
- Defend the protocol and claim boundary.
- Freeze a reviewable final release.
- Convert one limitation into a discriminating next experiment.
- Decide the current research stage.

## Slide 3: Week 13 made the artifact reviewable

Week 13 produced a fixed commit, independent reproduction record, focused correction, revised report, and bounded claim.

Week 14 selects and communicates the strongest evidence, releases the final package, and makes the next study testable.

## Slide 4: The talk is a selected research argument

$$
\text{problem}
\rightarrow
\text{question}
\rightarrow
\text{protocol}
\rightarrow
\text{evidence}
\rightarrow
\text{bounded conclusion}
\rightarrow
\text{next experiment}.
$$

A short talk is not a compressed report.

## Slide 5: Five questions the audience should answer

| Question | Evidence in the talk |
| --- | --- |
| What matters? | decision, population, or scientific gap |
| What was tested? | bounded question and prediction |
| How was it tested? | data, split, baseline, metric, selection |
| What happened? | central result and uncertainty |
| What follows? | limitation, release route, next experiment |

## Slide 6: Eight-minute evidence story

| Time | Purpose |
| ---: | --- |
| 0:00–0:45 | problem and relevance |
| 0:45–1:30 | question and contribution |
| 1:30–2:30 | data and evaluation |
| 2:30–4:15 | method and controlled comparison |
| 4:15–6:15 | main result and qualification |
| 6:15–7:15 | limitation |
| 7:15–8:00 | reproduction and next direction |

## Slide 7: Recommended main-slide sequence

1. title as question;
2. why the question matters;
3. what was tested;
4. data and evaluation protocol;
5. method or controlled comparison;
6. main evidence;
7. error, robustness, or negative evidence;
8. conclusion and limitation;
9. research direction;
10. reproduce and contact.

## Slide 8: State the contribution as a tested relation

> We tested whether [factor or method] changes [outcome] for [population/task] under [protocol], and found [result with uncertainty or boundary].

The contribution is the supported finding, analysis, or artifact—not merely the use of a model.

## Slide 9: Identify the evidence before the architecture

- observation and target;
- population and exclusions;
- data version and access;
- train, validation, and locked-test roles;
- baseline;
- metric and averaging;
- selection rule;
- source of variation.

## Slide 10: Explain only the method needed to interpret the result

| Keep in the main talk | Move to backup |
| --- | --- |
| controlled factor | complete layer listing |
| mechanism tied to hypothesis | full hyperparameter grid |
| baseline and fairness controls | installation details |
| selection and compute budget | secondary ablations |
| inference rule | full logs |

## Slide 11: The central result slide has seven parts

- conclusion-bearing title;
- one focused table or figure;
- population and split role;
- metric and direction;
- sample or seed count;
- uncertainty or observations;
- result ID or generating command.

## Slide 12: Name what varies

“Mean across five training seeds with a 95% Student-t interval” is more precise than “stable.”

Uncertainty may come from seeds, sampled observations, folds, sites, time periods, annotators, prompts, or hardware measurements.

## Slide 13: Negative evidence needs a capable test

Before interpreting no improvement, check:

- implementation;
- optimization and compute budget;
- baseline strength;
- metric sensitivity;
- sample size or repeats;
- split integrity; and
- whether the manipulation changed the intended mechanism.

## Slide 14: A limitation protects the conclusion

> Because [evidence boundary], the experiment does not establish [affected claim]. This may [consequence]. Test [specific experiment] using [decision rule].

## Slide 15: Use result titles

| Section label | Evidence-bearing title |
| --- | --- |
| Results | Group-wise splitting removed the apparent gain |
| Ablation | Removing the gate increased long-sequence error |
| Robustness | The benefit disappeared under site shift |
| Limitations | The result covers one site and five seeds |

## Slide 16: Visual evidence audit

- one question per figure or table;
- readable labels, units, counts, and uncertainty;
- direct comparison;
- no color-only distinction;
- declared example-selection rule;
- published and project results separated;
- source or generating command recorded.

## Slide 17: Design for the room and for access

Use high contrast, large type, consistent reading order, meaningful table headers, redundant labels and shapes, alt text in shared files, captions for media, a PDF fallback, and static backups for demos.

## Slide 18: Rehearsal is a communication experiment

1. deliver without interruption;
2. record finish time and first unclear point;
3. listener reconstructs question, result, and limitation;
4. compare intended and received message;
5. revise one title, transition, and removable detail;
6. repeat the opening, result, and conclusion.

## Slide 19: Classify the question before answering

| Question type | First response |
| --- | --- |
| clarification | define the object or protocol |
| evidence | point to result and uncertainty |
| validity | name threat and affected claim |
| alternative | separate observation from explanation |
| scope | state evaluated population |
| feasibility | name resources and approvals |
| future | name experiment and decision |

## Slide 20: Use the Q&A answer ladder

1. **Observation:** what was recorded.
2. **Interpretation:** what the protocol supports.
3. **Uncertainty:** what remains unidentified.
4. **Next evidence:** what would resolve it.

“I do not know” is useful when followed by a concrete evidence requirement.

## Slide 21: Build direction from an observed boundary

$$
\text{boundary}
\rightarrow
\text{competing explanations}
\rightarrow
\text{discriminating experiment}
\rightarrow
\text{decision rule}.
$$

Begin with the unresolved phenomenon, not a fashionable architecture.

## Slide 22: Match the direction to its current stage

| Stage | Current target |
| --- | --- |
| feasibility | verify access, measurement, baseline, and minimum signal |
| master's | complete one bounded contribution with available resources |
| PhD | develop a connected program of original questions and studies |

Local program expectations and supervision determine final scope.

## Slide 23: Plan the first 90 days

| Days | Decision evidence |
| ---: | --- |
| 1–30 | closest baseline, data access, literature gap, stop criteria |
| 31–60 | minimum controlled experiment and integrity checks |
| 61–90 | continue, narrow, redirect, or stop |

## Slide 24: Freeze the final release

- repository URL and fixed commit;
- clean working state;
- report and presentation source plus PDF;
- environment and data statement;
- final configuration and result command;
- expected output and agreement criterion;
- licenses, citation, limitations, and reproduction record.

## Slide 25: Static audit complements reproduction

```bash
python labs/week14_release_audit.py --audit-only

python labs/week14_release_audit.py \
  --project-root path/to/project \
  --manifest path/to/project/project-release.json
```

The audit checks declarations, paths, files, hashes, and release metadata. It does not execute project code.

## Slide 26: Final defense and exit ticket

Submit:

- fixed repository release;
- report and presentation;
- final-release checklist;
- Week 13 reproduction evidence;
- presentation self-assessment;
- research-direction note.

Exit ticket: What is established, what remains unresolved, and which experiment could change the conclusion?
