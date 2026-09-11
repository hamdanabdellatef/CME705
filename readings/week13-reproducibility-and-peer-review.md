# Week 13 further reading — Reproducibility, Reporting, and Peer Review

Use this guide to strengthen the project artifact and report. Begin with the three core sources, then choose readings that match the project's weakest evidence link.

## Core sources

### Reproducibility in machine learning

[Improving Reproducibility in Machine Learning Research](https://www.jmlr.org/papers/v22/20-303.html) reports the NeurIPS reproducibility program, including code submission, a reproducibility challenge, and an author checklist.

Read for:

- why reproducibility concerns both conducting and communicating research;
- how checklists can expose missing experimental detail;
- what community-scale reproduction efforts can and cannot establish.

Questions:

- Which checklist item would most improve your current report?
- Which requirement is expensive or impossible for your project, and how can you explain that limitation honestly?

### Artifact evaluation

The [ACM Artifact Review and Badging policy](https://www.acm.org/publications/policies/artifact-review-and-badging-current) distinguishes artifact availability, artifact quality, and validation of results. It also defines repeatability, reproducibility, and replicability for ACM review.

Read for:

- the difference between an available artifact and a functional, reusable artifact;
- why result validation requires an independent team;
- why terminology should identify the team, artifacts, and experimental setup.

Questions:

- Could another researcher obtain every required artifact?
- Which component would prevent reuse on a new dataset or system?

### Current author checklist

The [NeurIPS Paper Checklist guidelines](https://neurips.cc/public/guides/PaperChecklist) ask authors to address claims, limitations, experimental reproducibility, data and code access, training details, uncertainty, compute, ethics, licenses, and released assets.

Read for:

- how a claim's scope should match the experiments;
- what training and model-selection details belong in a report;
- how to define error bars and their source of variation;
- what compute and asset information a reviewer needs.

Questions:

- Can every “yes” answer point to a report section or artifact?
- Which “no” answer needs a justification and future plan?

## Research reporting

### Model cards

[Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596) proposes structured reporting for trained models, including intended use, evaluation conditions, limitations, and group-aware performance.

Use it when the project releases a trained model or checkpoint.

Audit:

- intended uses and excluded uses;
- training and evaluation data;
- metrics and disaggregated evidence;
- limitations and ethical considerations.

### Datasheets

[Datasheets for Datasets](https://doi.org/10.1145/3458723) proposes documentation for dataset motivation, composition, collection, preprocessing, uses, distribution, and maintenance.

Use it when data provenance, labels, consent, or future reuse materially affect the claim.

Audit:

- why and by whom the dataset was created;
- what one instance represents;
- who or what may be missing;
- how labels were obtained;
- what transformations and exclusions occurred;
- which uses may be inappropriate.

### Reporting standards for ML-based science

[REFORMS: Reporting Standards for Machine Learning Based Science](https://arxiv.org/abs/2308.07832) synthesizes reporting practices for studies that use machine learning to support scientific conclusions.

Use it when the project makes a domain-science claim rather than only a benchmark claim.

Questions:

- Is the target prediction scientifically meaningful?
- Does model evaluation test the scientific claim or only predictive performance?
- Could leakage, confounding, or site effects explain the result?

## Experimental credibility

### Underspecification

[Underspecification Presents Challenges for Credibility in Modern Machine Learning](https://arxiv.org/abs/2011.03395) shows that models with similar held-out performance can behave differently under stress or deployment-relevant tests.

Use it to design a robustness check that distinguishes models selected by the same validation metric.

Questions:

- Which behavior is left unspecified by the training objective and validation metric?
- Which stress test represents the intended use rather than arbitrary corruption?

### Troubling trends in scholarship

[Troubling Trends in Machine Learning Scholarship](https://jmlr.org/papers/v20/18-303.html) analyzes problems such as vague claims, weak explanation, and misuse of mathematics.

Use it to audit writing quality and contribution claims.

Questions:

- Does the report distinguish explanation from speculation?
- Does every equation clarify the method or evidence?
- Is the baseline strong enough to isolate the claimed contribution?

### Deep reinforcement learning reproducibility

[Deep Reinforcement Learning that Matters](https://ojs.aaai.org/index.php/AAAI/article/view/11694) studies sensitivity to random seeds, hyperparameters, implementations, and reporting choices in deep reinforcement learning.

The lessons extend beyond reinforcement learning: report seed-level evidence, tuning protocol, implementation details, and compute.

Questions:

- Which choices were tuned after observing validation or test results?
- Would a different implementation change the comparison?

## Software and artifact practice

### PyTorch reproducibility

The [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) explain random-number control, nondeterministic operations, deterministic algorithms, and performance tradeoffs.

Use the exact documentation version that matches the installed PyTorch release.

Record:

- Python, NumPy, and PyTorch seeds;
- deterministic-algorithm settings;
- backend flags;
- device and precision;
- operations known to remain nondeterministic.

### NumPy random generation

The [NumPy random generator documentation](https://numpy.org/doc/stable/reference/random/generator.html) describes explicit generator objects. Passing a generator makes random state visible and reduces dependence on global state.

Use:

```python
rng = np.random.default_rng(seed)
permutation = rng.permutation(indices)
```

Store the seed and code version. A seed without the generator and algorithm version may not reproduce indefinitely.

### Packaging and environments

The [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/) covers environments, dependency declarations, and package installation.

For the course artifact, provide:

- a clean-environment construction command;
- supported Python versions;
- declared dependencies;
- resolved versions for the final run;
- platform-specific accelerator instructions when needed.

A lock file records resolved software. It does not preserve data, hardware, external services, or licenses.

## Statistical reporting

### Name the variation source

For every error bar or interval, state what changed:

- training seed;
- split seed;
- bootstrap sample;
- participant sample;
- measurement repeat;
- site, device, or time period.

These sources support different claims. Five training seeds quantify algorithmic variation under one dataset and split. They do not quantify uncertainty across future populations.

### Prefer paired comparisons when justified

Pair conditions by the same seed, split, or observation when the pairing represents shared variation. Report paired differences:

$$
d_i=x_i^{(B)}-x_i^{(A)}.
$$

Then summarize $\bar{d}$ and $s_d$. State why the pairing is meaningful.

### Avoid metric-only conclusions

A small metric difference may have little practical meaning. A large difference may still result from leakage or an unfair compute budget. Connect the effect to:

- metric resolution and uncertainty;
- baseline strength;
- data and selection protocol;
- compute and parameter count;
- errors or subgroups;
- intended use.

## Peer-review practice

### Review the evidence, not the author

A useful review identifies a location, observed evidence, consequence, requested change, and verification plan. Avoid statements about competence or intent.

### Preserve the first attempt

Save commands, logs, versions, and errors before modifying the setup. An undocumented repair hides information needed to improve the artifact.

### Separate severity from effort

A one-line missing flag can invalidate a result. A difficult installation problem may have no effect on the scientific claim if the target can be verified another way. Explain the consequence directly.

### Verify the correction

A reply or code change does not resolve the issue by itself. Record the corrected commit and rerun the affected step.

## Reading paths

### Reproducible ML experiment

1. Pineau et al. reproducibility report
2. PyTorch or NumPy randomness documentation
3. ACM artifact review policy
4. NeurIPS experimental details and uncertainty checklist items
5. Week 13 lab and manifest

### Dataset-centered research

1. Datasheets for Datasets
2. NeurIPS assets, licenses, ethics, and human-subject checklist items
3. REFORMS
4. one domain-specific data standard
5. a project-specific access and split plan

### Model release

1. Model Cards
2. ACM functional and reusable artifact criteria
3. NeurIPS assets and safeguards items
4. framework reproducibility documentation
5. project-specific misuse and monitoring analysis

### Robustness and generalization

1. Underspecification
2. Deep Reinforcement Learning that Matters
3. REFORMS
4. one domain-shift benchmark or study
5. a project-specific stress test

### Scientific writing

1. Troubling Trends in Machine Learning Scholarship
2. NeurIPS claims and limitations items
3. the Week 13 report template
4. one strong paper in the project area
5. a revision comparing abstract, results, limitations, and conclusion

## Research-direction prompts

A future master's or PhD direction should emerge from observed evidence rather than model popularity.

### Reproducibility engineering

- Can an experiment specification generate portable runs across hardware and frameworks?
- How should provenance systems represent restricted data and external services?
- Which artifact checks predict successful independent reproduction?

### Evaluation validity

- When do common benchmark metrics fail to predict downstream utility?
- How stable are model rankings across feature extractors, subgroups, or sites?
- Which uncertainty sources dominate the final research conclusion?

### Data and distribution shift

- Which collection mechanism causes the observed performance gap?
- Can domain adaptation improve held-out-site performance without hiding subgroup failures?
- How should split construction represent the intended deployment boundary?

### Efficient and reliable training

- Which sources of nondeterminism materially change model selection?
- Can lower-precision or lower-compute training preserve the scientific conclusion?
- How should compute budgets enter fair model comparison?

### Human and institutional constraints

- How can researchers provide verifiable evidence when data cannot be shared?
- Which privacy-preserving summaries enable meaningful reproduction?
- How do consent and licensing constraints change artifact design?

## Deliverable

For one central project claim, cite:

1. one result file and generating command;
2. one source that motivates the method;
3. one source that motivates the evaluation or reporting protocol;
4. one limitation and an experiment that could change the conclusion.

Record the exact source version or access date. Explain why each source is relevant rather than listing it without connection to the project.