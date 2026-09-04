# Week 1 instructor guide

## Session purpose

Week 1 establishes the course language and the habit of connecting a method to an application decision, baseline, and evidence. The session uses a synthetic machine-maintenance example throughout. Students finish with a research-interest draft rather than a fixed thesis topic.

## Before class

- Verify the commands in [setup.md](../setup.md) in a fresh environment.
- Run `python labs/week01_python_numpy_diagnostic.py`.
- Print or distribute the [worksheet](../modules/01-ml-foundations/worksheet.md).
- Ask students to bring one application area that interests them.
- Decide whether students will work individually or in pairs during the diagnostic.

## Three-hour plan

| Time | Activity | Instructor action | Evidence |
| ---: | --- | --- | --- |
| 0–10 min | Opening case | Present the 20-inspection constraint and collect possible outputs | Students distinguish decision from prediction |
| 10–25 min | Course route | Explain the NumPy-to-PyTorch path and semester project | Students identify the Week 1 deliverables |
| 25–45 min | Programmed and learned solutions | Compare a temperature threshold with a fitted scoring model | Students name human choices in both approaches |
| 45–70 min | Vocabulary and task types | Work through observations, features, targets, and four task types | Students classify the four practice problems |
| 70–90 min | Problem canvas | Model one field, then let pairs complete the maintenance canvas | Each pair produces an unambiguous unit and target |
| 90–100 min | Break | Check environment issues | Students prepare the repository and terminal |
| 100–135 min | NumPy diagnostic | Ask for shape and outcome predictions before execution | Each student records outputs and one code change |
| 135–155 min | Results discussion | Separate execution evidence from application evidence | Students state one supported and one unsupported claim |
| 155–175 min | Research-direction ladder | Confer briefly with students and narrow broad topics | Each student drafts a task, baseline, and uncertainty |
| 175–180 min | Exit ticket | Collect three sentences from the worksheet | Application goal, learning task, required evidence |

## Teaching prompts

### Opening case

Ask:

- Does the team need a class, a risk score, a ranking, or a complete schedule?
- What outcome could be measured after the decision?
- Could the act of inspecting change the recorded target?

Expected insight: one application can support several ML formulations. The intended action and available labels determine which is useful.

### Unit of observation

Offer three choices: one machine, one reading, and one sensor window. Ask students what changes under each choice.

Expected insight: the unit determines the row structure and affects how data must be split. Repeated observations from one machine are dependent, which prepares students for group-aware splitting in Week 2.

### Target definition

Compare:

- technician recommends inspection;
- component fails within seven days;
- maintenance cost exceeds a threshold.

Expected insight: targets may encode human judgment, future events, or economic consequences. They are not interchangeable.

### Baseline

Ask why the majority-class rule can appear accurate when failures are rare.

Expected insight: a metric has meaning only relative to the class distribution, costs, intended use, and a reference method.

## Diagnostic facilitation

Students first predict:

- `features.shape`;
- the result shape of `features @ weights`;
- which feature has the largest raw weight; and
- whether the rule will beat the majority baseline.

After the first run, ask students to set one weight to zero and run again. Students should explain the result through the generated data process rather than claiming a general property of that feature.

The script uses generated labels from a noisy linear process. The candidate rule resembles that process, so it should outperform the majority baseline. State this openly. The activity checks Python, NumPy, shape reasoning, and interpretation. It is not a fair blind evaluation of model discovery.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| Deep learning is always more advanced and therefore better | Return to the application, available data, baseline, compute, and evidence |
| A target is an objective truth | Ask who created the label, when, and under what definition |
| Every row is independent | Use repeated readings from the same machine to show dependence |
| High accuracy proves practical usefulness | Ask about class balance, costs, new conditions, and the action |
| Clusters reveal natural categories | Ask how stability and application meaning would be tested |
| The lab result proves the rule works in a factory | List which parts of a real deployment the generated example omits |

## Formative feedback

Use three checks:

1. **Canvas check:** every field names a concrete object, time, or action.
2. **Code check:** the student can explain the shapes in `features @ weights`.
3. **Claim check:** the student states only what the diagnostic evidence supports.

Avoid grading programming speed. Record prerequisite gaps and provide a short refresh path before Week 4.

## Research-interest memo feedback

Strong drafts identify an affected system or population, an observable task, plausible data, and a specific difficulty. Redirect drafts that begin with a model name. Ask the student to define the decision or understanding first.

Do not require novelty in Week 1. The objective is a searchable and testable direction.

## Shorter delivery options

For a 120-minute class:

- assign the AI/ML/deep-learning distinction and NumPy quickstart before class;
- complete one task-identification example instead of four;
- run the diagnostic in pairs; and
- move the research-direction ladder to homework.

For an online class, use small groups for the problem canvas and require each group to post one ambiguous target definition and its revision.

## After class

- Review exit tickets for prerequisite gaps and target ambiguity.
- Group related research interests to guide future examples and optional readings.
- Confirm that each student has a feasible next step for the research-interest memo.
- Record recurring questions that should be added to the Week 1 notes.
