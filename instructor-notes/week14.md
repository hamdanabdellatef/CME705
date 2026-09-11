# Instructor guide — Week 14 Research Presentation and Final Release

## Purpose

Week 14 is the capstone defense and release session. Students should leave with a fixed, reviewable project package and a next experiment that follows from evidence. The session evaluates reasoning, communication, reproducibility, and research judgment rather than architectural complexity.

## Instructor preparation

Before class:

1. announce the talk and question times;
2. decide how overflow presentations will be scheduled;
3. publish the final submission channel and deadline;
4. test the projection system, PDF fallback, and timing display;
5. prepare printed or digital presentation rubrics and peer-feedback forms;
6. verify accessibility arrangements;
7. run `python labs/week14_release_audit.py --audit-only`;
8. remind students to remove credentials and restricted data;
9. state whether repositories must be public, private to the instructor, or institutionally archived; and
10. define the policy for recording presentations.

Do not require public release when data, consent, intellectual-property, or security conditions prohibit it. Require a reviewable permitted package instead.

## Recommended room setup

- visible countdown timer;
- presenter computer tested with PowerPoint and PDF;
- static image backup for live demos;
- seating that supports questions without interrupting recording or access;
- private route for feedback containing sensitive research concerns;
- one rubric per presenter;
- one peer-feedback form per student.

## Suggested session plan

| Time | Activity | Instructor role |
| --- | --- | --- |
| 0–15 min | release gate | verify commit, status, report, deck, and manifest |
| 15–35 min | evidence-story mini lesson | model a bounded claim and sequence |
| 35–55 min | result-slide audit | compare a section title with an evidence title |
| 55–75 min | paired rehearsal | enforce uninterrupted first run and observation record |
| 75–165 min | presentation block | keep time and collect rubric evidence |
| 165–180 min | break | verify next presenters' files |
| 180–200 min | direction clinic | test competing explanations and feasibility |
| 200–210 min | synthesis | collect sign-off and exit ticket |

Repeat the presentation block or schedule another session when enrollment requires it.

## Presentation timing

Recommended per student:

- 8 minutes presentation;
- 4 minutes questions;
- 2–3 minutes transition and rubric completion.

Give a visible warning at 6:30 and stop the talk at 8:00 under the announced policy. Apply the same timing rule to all students. Accessibility accommodations take priority.

## Opening prompt

Display a result such as “accuracy = 0.81” and ask:

- What would you need to believe this number?
- What should appear in the talk?
- What can remain in the released artifact?
- What would make the claim broader than the evidence?

Use answers to separate the presentation story from the audit trail.

## Teaching the evidence story

Write:

```text
problem -> question -> protocol -> evidence -> bounded conclusion -> next experiment
```

Ask students to locate each object in their current deck. If two consecutive slides do not have a clear relationship, require a spoken transition or remove one.

A contribution should name what was learned, built, measured, or made reproducible. “We used a CNN” is a method statement. “Grouped splitting removed the apparent gain” is a finding.

## Result-title exercise

Give pairs three headings:

- Results
- Model comparison
- Limitations

Ask them to rewrite each as a sentence supported by their own evidence. Then ask which visible object allows the audience to verify the title.

Watch for:

- missing population;
- validation result presented as final test evidence;
- mean without variation;
- “significant” without a declared analysis;
- broad “better” or “robust” claims;
- causality from an uncontrolled comparison.

## Visual audit

Project one dense report table and ask what the talk needs. A good revision may:

- keep only baseline and selected model;
- retain one metric aligned with the question;
- show seed-level points or uncertainty;
- mark test status;
- add sample count;
- use a title stating the observed comparison;
- move full values to backup.

Do not reward decoration. Reward faster and more accurate interpretation.

## Rehearsal rules

The first listener does not interrupt. They record:

- finish time;
- first unclear point;
- research question heard;
- central result heard;
- limitation heard.

This prevents the speaker from verbally repairing the deck before observing the communication failure.

After the reconstruction, allow five minutes for revision and a three-minute rerun of opening, main result, and conclusion.

## Presentation assessment

Use [the presentation rubric](../research-project/presentation-rubric.md). Record evidence during the talk. Do not infer missing protocol details from earlier conversations with the student.

Evaluate:

- question and contribution;
- experimental protocol;
- evidence quality;
- interpretation and limitations;
- reproducibility and release;
- visual and oral communication;
- Q&A defense;
- research direction.

Model complexity receives no independent credit.

## Question design

Use questions that test reasoning:

### Protocol

- Which data were available during model selection?
- What unit was split?
- How was the baseline selected?
- What remained fixed across the comparison?

### Evidence

- What does each point or row represent?
- What source of variation does the interval cover?
- How many candidate configurations were inspected?
- Which output file produced this figure?

### Validity

- Which alternative explanation remains?
- Which population is not covered?
- How would the result change under a different threshold?
- What conclusion would be invalid if preprocessing leaked information?

### Research direction

- What predictions differ under your two explanations?
- What is the smallest informative experiment?
- What result would make you stop?
- Which data, skill, compute, collaborator, or approval is currently missing?

Do not turn Q&A into a display of obscure trivia.

## Answer-quality guide

Strong answers:

1. state the observation;
2. interpret within the protocol;
3. name uncertainty or limitation;
4. propose the required evidence.

A student may correctly answer that the current project cannot resolve the question. Give credit when the boundary and next test are precise.

Weak answers:

- invent values or citations;
- replace evidence with confidence;
- broaden from one dataset to a field;
- label any future use of a new architecture as research;
- avoid a real limitation.

## Managing negative results

Ask whether:

- implementation checks passed;
- the baseline was credible;
- the budget could reveal the effect;
- the metric could detect it;
- the comparison was controlled;
- uncertainty is visible.

If not, the honest conclusion may be inconclusive. This can still support a strong research plan.

## Release gate

Before accepting the final artifact, confirm:

- repository URL and fixed commit;
- clean or explicitly documented working state;
- report and presentation open correctly;
- central command and expected output;
- data source, access, license, and split;
- environment and configuration;
- Week 13 review correction;
- licenses and citation;
- privacy and security review;
- research-direction note.

The Week 14 audit is static. It checks declarations and hashes but does not execute the reproduction command. Week 13 provides independent execution evidence.

## Running the release audit

Demonstrate:

```bash
python labs/week14_release_audit.py --audit-only
```

For a student project:

```bash
python labs/week14_release_audit.py \
  --project-root path/to/project \
  --manifest path/to/project/project-release.json
```

Optional Git checks should be run only from the intended project root:

```bash
python labs/week14_release_audit.py \
  --project-root path/to/project \
  --manifest path/to/project/project-release.json \
  --verify-git
```

Review any failure before encouraging a student to change files. A mismatch can indicate a stale manifest, wrong root, tampered artifact, or uncommitted correction.

## Research-direction clinic

Require two plausible explanations before accepting the next experiment. Ask what each explanation predicts.

A useful direction has:

- an observed boundary;
- an enduring question;
- a minimum discriminating experiment;
- a decision rule;
- feasibility evidence;
- a fallback;
- a contribution if the primary prediction fails.

A model name without a hypothesis is an implementation choice.

## Master's and PhD discussions

Avoid promising admission, supervision, funding, publication, or degree suitability. Program requirements differ.

Use the planning distinction:

- **feasibility:** establish access, measurement, and minimum signal;
- **master's:** one bounded contribution with available supervision and resources;
- **PhD:** a connected multi-study program with an enduring gap.

Recommend that students discuss the direction with prospective supervisors and verify institutional requirements.

## Feedback protocol

Peer feedback should describe the research communication:

- question understood;
- strongest evidence;
- overbroad claim;
- unresolved artifact question;
- informative next experiment.

Prohibit comments on accent, personality, appearance, or presentation style unrelated to clarity and access.

Instructor feedback should cite an observed slide, statement, answer, or artifact.

## Academic integrity and automated tools

Apply the published cohort policy. Ask students to identify external code, generated text, generated figures, pretrained models, datasets, and substantial assistance where required.

Assessment should focus on whether the student can explain, trace, and defend every central component. A student remains responsible for incorrect or fabricated material regardless of the tool used.

## Accessibility

Provide alternatives for students who cannot present in the default format. Possible adjustments include:

- extended time;
- recorded presentation with live questions;
- poster or structured oral examination;
- assistive technology;
- captioned media;
- accessible digital documents.

Preserve the same learning outcomes while following institutional accommodations.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| more slides show more work | the talk selects the evidence needed for one argument |
| a live demo proves the method | a demo shows one execution; the release and evaluation support claims |
| a clean audit proves validity | the audit checks declared package consistency |
| a negative result is a failed project | a checked negative result can constrain a hypothesis |
| limitations weaken the defense | precise limitations make the conclusion credible |
| future work means trying a larger model | research direction requires a question and discriminating experiment |
| a tag is automatically immutable | record the commit and verify release assets |

## Submission handling

Use the institutional submission route. Do not require students to email large archives when a managed repository or learning platform is available.

Record:

- submission timestamp;
- commit or archive hash;
- access status;
- report and presentation versions;
- audit result;
- accommodation or extension where applicable.

Avoid downloading student repositories into the public course repository.

## Course closing discussion

Ask each student to complete:

1. The strongest claim I can defend is…
2. The most important unresolved claim is…
3. The experiment that could change my conclusion is…
4. The skill, data, compute, collaborator, or approval I need next is…

Connect answers to the progression of the course: formulation, data, evaluation, computation, optimization, generalization, architectures, generative modeling, reproducibility, and research practice.

## After class

- verify access to every submission;
- preserve grading records privately;
- return rubric evidence through the approved channel;
- do not publish student work without permission;
- record broken instructions or tools for the next course release;
- update links and software versions before the next offering.

## Instructor checklist

- [ ] timing and transition policy announced;
- [ ] overflow plan prepared;
- [ ] projection and PDF fallback tested;
- [ ] rubric and peer forms distributed;
- [ ] accessibility arrangements implemented;
- [ ] release-audit example passed;
- [ ] each talk has a recorded commit;
- [ ] central evidence and limitation assessed;
- [ ] Q&A tested reasoning;
- [ ] research direction includes a decision rule;
- [ ] submissions preserved through the institutional route;
- [ ] student work remains private unless permission is explicit.
