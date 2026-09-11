# Week 14 notes — Defending Evidence and Designing the Next Study

## 1. The capstone is a research argument

The final week combines communication, release engineering, and research planning. These activities are connected. A result that cannot be traced is difficult to defend. A polished talk with an unsupported conclusion is still weak research. A well-documented limitation can become the most valuable part of a future thesis plan.

The audience should leave with one central statement:

> Under a declared protocol and population, the project observed a specific result that supports a bounded conclusion and motivates a testable next experiment.

## 2. Select one contribution

A semester project may contain many experiments. A short talk should select the contribution with the clearest evidence chain.

A contribution may be:

- an empirical finding under a new or relevant setting;
- a controlled comparison of two methods;
- a useful negative result;
- an evaluation or data-quality finding;
- a reproducible implementation and diagnostic analysis;
- a robustness or failure-mode study; or
- a feasible research question supported by a strong pilot.

It is rarely “we used model X.” The model is part of the method. The contribution is what the evidence adds to understanding or practice.

### Contribution sentence

Use:

> We tested whether [factor or method] changes [outcome] for [population/task] under [protocol], and found [result with uncertainty or boundary].

Remove claims that are not used in the rest of the talk.

## 3. Audience and scope

Assume an audience that understands machine-learning fundamentals but has not followed the semester project. Define domain terms, observation units, targets, and metrics before relying on them.

A short talk needs less background than a report. Include only the context required to understand:

- the decision or scientific problem;
- the gap;
- the comparison;
- the metric; and
- the conclusion.

Move literature inventories to the report. Use one or two primary references on slides where they support a design choice or comparison.

## 4. Story structure

The research story is a sequence of questions:

1. Why does the problem matter?
2. What is unknown or unreliable?
3. What did the project test?
4. How was the test protected from leakage and unfair comparison?
5. What happened?
6. What does the result establish?
7. What remains unresolved?
8. What should be tested next?

Every slide should advance this sequence. A slide that does not change the audience's understanding may belong in backup material.

## 5. Timing as an experimental constraint

An eight-minute talk is a fixed budget. Plan the budget before designing slides.

| Segment | Seconds | Role |
| --- | ---: | --- |
| problem and gap | 45 | establish relevance |
| question and contribution | 45 | bound the claim |
| data and evaluation | 60 | define evidence |
| method and comparison | 105 | explain only necessary mechanism |
| main result and qualification | 120 | show result, uncertainty, and errors |
| limitation | 60 | protect claim scope |
| direction and reproducibility | 45 | make continuation and verification concrete |

Allow a margin for transitions and projection delay. Practice under the actual display format.

## 6. Slide titles as claims

Section labels such as “Method,” “Results,” and “Discussion” do not tell the audience what matters.

Prefer titles such as:

- “Participant-wise splitting removed the apparent 12-point gain.”
- “The LSTM retained the signal at lengths unseen during selection.”
- “The VAE improved sample coverage but reduced reconstruction sharpness.”
- “The observed benefit is limited to one site and five training seeds.”

The slide body should let the audience verify the title. If it cannot, narrow the title or add the missing evidence.

## 7. Designing the central result slide

A strong central result slide contains:

1. a conclusion-bearing title;
2. a figure or table focused on one comparison;
3. sample count, split role, metric, and direction;
4. uncertainty or seed-level observations when meaningful;
5. a short annotation of the practical difference;
6. the generating command or result ID in the notes or footer; and
7. a spoken statement of what the result does not establish.

### Tables

Use a table when exact values and protocol fields matter. Limit columns to the comparison needed. Align decimals and mark whether values are mean, spread, interval, or one run.

### Charts

Use a chart when pattern, distribution, trajectory, or uncertainty matters. Show observations when the sample is small. Avoid three-dimensional effects, decorative gradients, truncated axes without justification, and color-only categories.

### Images and qualitative examples

State how examples were selected. Do not present hand-picked successes as performance evidence. Pair examples with a declared sampling rule and quantitative result.

## 8. Uncertainty in speech and graphics

Name the source of variation:

- training seeds;
- data samples;
- folds;
- sites;
- time periods;
- annotation choices;
- prompts; or
- hardware measurements.

Say “mean across five training seeds, with the 95% Student-t interval” rather than “the model is stable.” The interval describes a declared source of variation under a fixed protocol. It does not automatically describe population, site, label, or implementation uncertainty.

When there are too few runs for a useful interval, show the individual runs and state the limitation.

## 9. Negative and mixed evidence

A negative result can reject a directional prediction only when the experiment could reasonably reveal the expected difference.

Audit:

- implementation checks;
- optimization and resource budget;
- baseline strength;
- metric sensitivity;
- sample size or repeated runs;
- data and split integrity; and
- whether the manipulation actually changed the intended mechanism.

If these remain uncertain, say the experiment was inconclusive and propose a diagnostic test.

Mixed evidence may support a tradeoff:

> The treatment improved macro recall but increased calibration error and compute.

Do not compress conflicting metrics into an unweighted claim of superiority.

## 10. Limitations as part of the result

A limitation should specify:

- the evidence boundary;
- the affected claim;
- the likely consequence;
- a plausible competing explanation; and
- the next check.

Template:

> Because [boundary], the experiment does not establish [claim]. This may [consequence]. Compare [experiment] using [decision rule].

Limitations should appear before the final conclusion, not only in response to questions.

## 11. Accessible presentation design

Use:

- high contrast and a consistent reading order;
- large type readable from the back of the room;
- direct labels rather than distant legends when possible;
- redundant shape, label, or line style in addition to color;
- concise alternative text in shared digital copies;
- captions or transcripts for video and audio;
- meaningful table headers;
- a PDF fallback; and
- static backup images for animations or live demos.

Avoid dense screenshots. Recreate the relevant information as a readable diagram or crop with context and source.

## 12. Ethical communication

Presentation constraints do not justify omitting evidence that changes interpretation.

Disclose:

- relevant subgroup failures;
- unavailable or restricted data;
- material preprocessing decisions;
- test-set reuse;
- selection after seeing results;
- external services and model versions;
- conflicts, permissions, or consent constraints; and
- known risks or inappropriate uses.

Anonymize peer feedback and do not display private student or participant records.

## 13. Rehearsal as measurement

Record more than whether the talk “felt good.”

| Observation | Possible response |
| --- | --- |
| audience states the wrong question | rewrite opening question and transition |
| main result is remembered without protocol | add population, split, and metric to title or annotation |
| timing exceeds limit | remove one background or method detail |
| limitation sounds like an apology | connect it to the affected claim and experiment |
| audience cannot read a chart | simplify, enlarge, or split the figure |
| answer becomes speculative | use observation–inference–uncertainty–next-test structure |

Use at least one listener who has not worked on the project.

## 14. The Q&A answer ladder

A defensible answer can move through four levels:

1. **Observation:** what the project recorded.
2. **Interpretation:** what the protocol supports.
3. **Uncertainty:** what remains variable or unidentified.
4. **Next evidence:** the experiment or data needed.

Example:

> We observed a positive paired difference across five seeds on the locked split. That supports a benefit under this configuration. It does not identify whether the interaction helps across sites or acquisition settings. A site-held-out evaluation with the same selection rule would test that scope.

Do not begin with a broad explanation when the question is asking for an observed value.

## 15. Handling difficult questions

### The question assumes a result you did not test

Correct the premise respectfully and state the nearest supported result.

### The question identifies a real flaw

Acknowledge the consequence, narrow the claim, and describe the smallest corrective experiment.

### The question proposes another method

Explain whether it is a baseline, competing mechanism, or future extension. State what comparison would be fair.

### The answer is unknown

Say what is unknown and what evidence would resolve it. Avoid inventing a citation, number, or experiment outcome.

## 16. Backup slides

Prepare backup slides for:

- split construction and counts;
- architecture and parameter count;
- full hyperparameters;
- training and validation curves;
- confusion matrix or error examples;
- additional seed-level results;
- robustness checks;
- compute and environment;
- license or data-access constraints; and
- reproduction commands.

Backup slides are part of the reviewed presentation artifact. Label exploratory analyses honestly.

## 17. Release identity

A final release should be identifiable by repository URL and immutable commit. A tag helps communication, but the commit is the primary state identifier.

Before freezing:

```bash
git status --short
git rev-parse HEAD
python -m pytest
```

Record generated artifacts separately from source. Do not commit caches, downloaded datasets, model weights, or private data unless release rights and repository policy explicitly permit them.

## 18. Final package anatomy

A reviewer needs:

- a landing README;
- installation instructions;
- supported software and hardware;
- data provenance and access;
- configuration actually used;
- commands for the central result;
- expected output and agreement criterion;
- report and presentation;
- license and citation;
- artifact hashes or release manifest;
- known limitations; and
- contact or issue route where appropriate.

The package should not depend on verbal context from the presentation.

## 19. Hashes and manifests

A SHA-256 value detects byte changes. It does not establish that a file is correct, ethical, licensed, or scientifically valid.

Use a manifest to identify central files:

```json
{
  "path": "results/final_metrics.json",
  "sha256": "64 hexadecimal characters"
}
```

Use relative paths. Exclude credentials, absolute personal paths, and private URLs.

## 20. Release audit behavior

The Week 14 audit performs static checks. It does not execute a command from a student manifest.

It verifies:

- schema and required sections;
- safe relative paths;
- declared artifact existence;
- valid and matching SHA-256 values;
- commit format and optional repository-state agreement;
- data access, license, and split statements;
- expected result and agreement criterion;
- privacy review and known limitations; and
- common credential-like patterns.

A passing audit means the declared package is internally consistent. It does not replace the Week 13 independent reproduction.

## 21. Research direction as a causal question

Begin with an observed boundary, not an architecture name.

Weak:

> Future work will use a Transformer.

Stronger:

> Performance falls when relevant evidence occurs beyond the training sequence lengths. Compare a recurrent, attention, and state-space model under matched parameters and compute. The direction is supported if the long-context slope improves without degrading in-range calibration.

The stronger direction names the phenomenon, competing mechanisms, controlled comparison, and decision.

## 22. Competing explanations

Suppose a model underperforms on one subgroup. Plausible explanations might include:

- representation mismatch;
- label-quality differences;
- population shift;
- insufficient samples;
- calibration differences;
- data leakage in the apparent reference performance; or
- a threshold selected for the majority group.

The next experiment should produce different predictions under at least two explanations. Otherwise it adds data without resolving the question.

## 23. Minimum discriminating experiment

Specify:

- one independent variable;
- fixed controls;
- sampling and split design;
- baseline;
- outcome and metric;
- uncertainty unit;
- decision threshold;
- resource ceiling;
- stop condition; and
- interpretation for positive, negative, and inconclusive outcomes.

A minimum experiment is small enough to perform early and informative enough to determine whether the direction should continue.

## 24. Feasibility and approvals

List:

- data access and ownership;
- annotation or domain expertise;
- compute, memory, storage, and time;
- required mathematical or engineering skills;
- collaborators;
- ethics or institutional review;
- privacy and security controls;
- licensing and publication constraints; and
- fallback data or synthetic audit route.

A direction can be scientifically interesting and currently infeasible. Separating these judgments improves planning.

## 25. Master's planning

A master's direction usually benefits from:

- one bounded research question;
- a feasible dataset and baseline;
- one main controlled comparison;
- one robustness or external-validity check;
- a deliverable system, analysis, or method;
- a 6–18 month staged plan; and
- a contribution that remains useful if the headline prediction fails.

Local program regulations and supervision determine the final scope.

## 26. PhD planning

A PhD direction should support a connected program rather than one experiment.

Look for:

- an enduring gap;
- several linked hypotheses;
- a methodological, theoretical, systems, or domain contribution;
- multiple datasets, populations, or experimental settings;
- opportunities for independent publications;
- a progression from measurement to mechanism to generalization; and
- a reason the program remains valuable if one study is negative.

A first-year plan should still begin with a minimum discriminating experiment.

## 27. A 90-day research start

### Days 1–30: verify the gap

- reproduce the closest baseline;
- confirm data access and measurement;
- refine the literature map;
- define success, failure, and stop criteria.

### Days 31–60: run the minimum experiment

- implement one controlled comparison;
- verify evaluation integrity;
- record uncertainty and compute;
- diagnose the first discrepancy.

### Days 61–90: decide

- continue, narrow, redirect, or stop;
- document the decision evidence;
- identify the next skill, collaborator, data source, or approval;
- prepare a one-page research brief.

## 28. Final defense standard

A strong final submission allows a reader to:

- locate the question and contribution;
- distinguish observation from inference;
- trace every central result;
- understand uncertainty and limitations;
- reproduce one result from the fixed release;
- see why the next experiment is informative; and
- determine whether the proposed direction is feasible.

Polish matters because it reduces ambiguity. It cannot substitute for evidence.

## 29. Course synthesis

The course moved from defining observations and targets to building neural computations, training them, evaluating generalization, studying spatial and sequential structure, modeling distributions, and releasing a defensible research artifact.

The recurring discipline is the same:

1. define the question;
2. declare the protocol;
3. implement and check;
4. observe;
5. compare;
6. limit the claim; and
7. design the next test.

## 30. Claims supported by Week 14

After completing the package, a student can claim that they:

- presented one project contribution within a fixed time budget;
- linked the central result to a declared protocol and artifact;
- answered questions without expanding beyond the evidence;
- released a statically audited final package;
- identified a limitation and competing explanation; and
- designed a feasible next experiment.

The week does not establish that every project is publication-ready, generalizes beyond its evaluated setting, or already forms a complete thesis proposal.
