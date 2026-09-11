# Week 14 — Research Presentation, Final Release, and Next Direction

**Guiding question:** How can a researcher present one defensible contribution, release the evidence behind it, and turn an unresolved limitation into a feasible next study?

Week 14 completes the semester project. Students present a concise evidence-centered argument, defend the protocol and claim boundaries during questions, freeze a reviewable final release, and write a research-direction note that can support a master's thesis, PhD proposal, or applied research plan.

## Learning outcomes

By the end of the session, students should be able to:

- organize a short research talk around one question and one supported contribution;
- make every result slide state a conclusion that the displayed evidence supports;
- explain the population, split, baseline, metric, uncertainty, and compute behind a central result;
- distinguish a negative result from an uninformative experiment;
- answer questions by separating observation, interpretation, uncertainty, and next evidence;
- release a fixed repository state with report, presentation, commands, data statement, licenses, and citation;
- convert a specific limitation into a discriminating next experiment; and
- judge whether a direction is currently suitable for a master's project, a PhD program, or further feasibility work.

## Required preparation

Bring:

1. the verified Week 13 project commit and reproduction record;
2. an 8-minute talk with 8–10 main slides and optional backup slides;
3. the revised project report;
4. a completed [final-release checklist](../../research-project/final-release-checklist.md);
5. a draft [research-direction note](../../research-project/research-direction-note.md); and
6. a copy of the [presentation rubric](../../research-project/presentation-rubric.md).

Remove credentials, private records, restricted files, machine-specific secrets, and generated caches before release.

## Suggested session plan

The presentation block can be repeated when enrollment requires more than six speakers.

| Time | Activity | Evidence produced |
| --- | --- | --- |
| 0–15 min | Final release gate | Fixed commit, clean status, and required artifacts |
| 15–35 min | Evidence-centered talk structure | One-sentence question and contribution |
| 35–55 min | Result-slide and visual audit | One revised evidence slide |
| 55–75 min | Timed rehearsal in pairs | Timing record and one actionable revision |
| 75–165 min | Six presentations, 8 minutes plus 4 minutes of questions | Rubric and peer feedback |
| 165–180 min | Break or transition | — |
| 180–200 min | Research-direction clinic | Discriminating next experiment |
| 200–210 min | Course synthesis and submission freeze | Final sign-off |

## The talk contract

A short research presentation should let the audience answer five questions:

1. What problem and population are being studied?
2. What precise question or hypothesis was tested?
3. What protocol produced the evidence?
4. What conclusion does the evidence support, with what uncertainty and limits?
5. What experiment should happen next?

The presentation is not a compressed report. It is a selected argument. Details needed for audit but not for the main story belong in backup slides and the released artifact.

## An eight-minute evidence story

| Time | Purpose | Minimum content |
| ---: | --- | --- |
| 0:00–0:45 | Motivate | Decision, population, and why the problem matters |
| 0:45–1:30 | Bound | Research question, contribution, and claim scope |
| 1:30–2:30 | Identify evidence | Data, split, baseline, and metric |
| 2:30–4:15 | Explain protocol | Method, controlled factor, selection rule, and compute |
| 4:15–6:15 | Show result | Main result, uncertainty, error pattern, or robustness check |
| 6:15–7:15 | Limit | Threat to validity and conclusion not established |
| 7:15–8:00 | Continue | Reproduction route and next discriminating experiment |

Practice to finish the planned conclusion before time expires. Do not accelerate through limitations or the next experiment to compensate for an overlong introduction.

## Recommended slide sequence

Use 8–10 main slides:

1. **Title as question:** project title, author, affiliation, and repository or release identifier.
2. **Why this question matters:** intended use, affected population, and practical or scientific gap.
3. **What was tested:** bounded question, directional prediction, and contribution.
4. **Data and evaluation protocol:** observation, target, split, baseline, metric, and selection rule.
5. **Method or controlled comparison:** only the mechanism needed to interpret the result.
6. **Main evidence:** table or figure with uncertainty and an evidence-bearing title.
7. **Errors, robustness, or negative evidence:** one result that qualifies the main conclusion.
8. **Supported conclusion and limitation:** claim boundary and threat to validity.
9. **Research direction:** competing explanations, next experiment, and decision rule.
10. **Reproduce and contact:** fixed release, command, expected output, and citation.

Place architecture details, hyperparameters, full confusion matrices, extra examples, ablations, and environment records in backup slides when they are useful for questions.

## Evidence-bearing titles

A heading such as “Results” names a section. A heading such as “The interaction feature improved paired accuracy across all five seeds” states the result the audience should inspect.

For every evidence slide, complete:

> Under [protocol and population], [observed comparison] with [uncertainty or variation], supporting [bounded claim].

The title should not claim causality, generality, or superiority beyond the design.

## Visual evidence audit

Every chart or table should answer one question. Check:

- the compared conditions are named;
- axes, units, classes, and sample counts are visible;
- uncertainty is defined;
- the test or validation role is clear;
- color is not the only carrier of meaning;
- text remains readable when projected;
- the source or generating command is recorded;
- published and project-generated results are visually distinct; and
- decorative elements do not compete with the result.

A result table should emphasize the comparison that matters. A figure should display observations or summaries that the speaker explicitly interprets.

## Presenting a negative result

A negative result is informative when:

- the implementation and protocol passed their checks;
- the comparison had a credible chance to detect the expected effect;
- the resource budget is stated;
- uncertainty or variability is visible;
- plausible explanations are separated; and
- a smaller next experiment can distinguish them.

Do not rename a failed run as evidence against a method when optimization, data access, implementation, or statistical power remains unresolved.

## Rehearsal protocol

Work in pairs:

1. The speaker delivers the full talk without interruption.
2. The reviewer records the finish time and the first point where the argument became unclear.
3. The reviewer states the research question, main result, and limitation as understood.
4. The speaker compares that reconstruction with the intended message.
5. Revise one title, one transition, and one removable detail.
6. Deliver the opening, central result, and conclusion again in three minutes.

Rehearsal is an experiment on communication. Record the observed misunderstanding before changing the deck.

## Question-and-answer protocol

Listen for the type of question:

| Question type | First response |
| --- | --- |
| clarification | define the object, population, metric, or protocol |
| evidence | point to the result, uncertainty, and generating procedure |
| validity | name the affected claim and the threat |
| alternative explanation | distinguish what is observed from what remains plausible |
| generalization | state the evaluated population and proposed external test |
| feasibility | state data, compute, skills, time, and approvals |
| future work | name the experiment and result that would change the conclusion |

A strong answer can be:

> The current experiment does not establish that. It shows [observation] under [protocol]. The smallest test of your explanation would be [experiment], using [decision rule].

“I do not know” is acceptable when followed by what evidence would be needed.

## From limitation to research direction

Use this chain:

$$
\text{observed boundary}
\rightarrow
\text{competing mechanisms}
\rightarrow
\text{discriminating experiment}
\rightarrow
\text{decision rule}
\rightarrow
\text{research program}.
$$

A research direction should contain:

- an observed limitation or unexplained pattern;
- at least two plausible explanations;
- one experiment that makes different predictions under those explanations;
- a metric and result that would change the current conclusion;
- required data, compute, methods, collaborators, and approvals;
- a feasible first stage; and
- a reason the question matters beyond one dataset or implementation.

## Master's, PhD, or feasibility stage?

Program expectations vary by university and discipline. Use the categories as planning prompts.

| Direction | Typical shape | Evidence needed now |
| --- | --- | --- |
| feasibility study | establish access, measurement, baseline, and minimum signal | a bounded pilot and explicit stop condition |
| master's direction | one focused contribution completed with available supervision and resources | tractable question, usable data, baseline, and 6–18 month experiment plan |
| PhD direction | a connected program of original questions with several publishable studies | enduring gap, multiple hypotheses, methodological or scientific depth, and a staged multi-year plan |

A fashionable architecture is not a research direction. The direction comes from an unresolved question and an experiment capable of changing current understanding.

## Final release package

The final submission should identify:

- repository URL and fixed commit or release tag;
- report source and final PDF;
- presentation source and exported PDF;
- environment construction and observed package versions;
- data provenance, version, access, license, and split identity;
- final configuration, seeds, compute, and model-selection rule;
- one reproduction command, expected output, and agreement criterion;
- manifest or hashes for central artifacts;
- Week 13 peer-reproduction result and corrected commit;
- project license and citation information;
- privacy, ethics, security, and redistribution decisions;
- known limitations; and
- research-direction note.

A Git tag or hosted release is useful only when it resolves to the reviewed commit and the assets match the submission.

## Final release audit

Use the portable audit without executing project code:

```bash
python labs/week14_release_audit.py --audit-only
python labs/week14_release_audit.py \
  --project-root path/to/project \
  --manifest path/to/project/project-release.json
```

Copy [the example manifest](../../research-project/project-release.example.json), replace every placeholder, add SHA-256 values for the declared artifacts, and run the audit from a clean repository. The audit checks structure, safe paths, declared files, hashes, release metadata, and accidental credential patterns. It deliberately does not execute the reproduction command.

## Deliverables

Complete [Assignment 6](../../assignments/06-final-project-submission.md):

- final fixed repository release;
- 5–7 page report;
- 8-minute presentation and backup slides;
- completed final-release checklist;
- Week 13 reproduction record;
- research-direction note; and
- author self-assessment against the final-project and presentation rubrics.

## Week 14 package

- [Detailed notes](week14-notes.md)
- [Accessible slide source](week14-slides.md)
- [Student worksheet](week14-worksheet.md)
- [Instructor guide](../../instructor-notes/week14.md)
- [Instructor PowerPoint](slides/week14-research-presentation-and-final-release.pptx)
- [Student presentation template](../../research-project/slides/cme705-project-presentation-template.pptx)
- [Final-project submission](../../assignments/06-final-project-submission.md)
- [Presentation rubric](../../research-project/presentation-rubric.md)
- [Peer-feedback form](../../research-project/peer-feedback.md)
- [Research-direction note](../../research-project/research-direction-note.md)
- [Final-release checklist](../../research-project/final-release-checklist.md)
- [Release-audit lab](../../labs/week14_release_audit.py)
- [Further reading](../../readings/week14-research-communication-and-release.md)

## Exit ticket

Answer in three sentences:

1. What is the strongest claim established by the project?
2. What important claim remains unresolved?
3. Which next experiment could change the current conclusion?
