# Assignment 3: Literature comparison

## Purpose

Map the evidence surrounding the proposed research question. The comparison should reveal what has already been tested, which results can be compared fairly, which baseline is credible, and what question remains feasible for the semester project.

This is an evidence-analysis assignment rather than a list of paper summaries.

## Deliverables

Submit:

1. a completed [literature comparison table](../research-project/literature-review-template.md) covering at least five credible sources; and
2. a 700–1,000 word synthesis, excluding references and the table.

Use a clear filename such as student-id_literature-comparison.pdf. Provide persistent links or DOIs where available.

## Source requirements

- Prefer peer-reviewed primary research papers that report an implemented method and evaluation.
- Include the paper that best supports the simplest credible baseline.
- Include dataset documentation when a dataset's collection or split determines comparability.
- Use a review or tutorial to locate primary sources, but do not let it replace all primary evidence.
- Avoid treating vendor pages, unsourced blog posts, or model leaderboards as independent scientific evidence.

If the topic is emerging and five directly relevant papers do not exist, document the search and include the closest methodological or application evidence. Explain the gap rather than stretching relevance.

## 1. Define the search scope

Record:

- the research question;
- databases or search systems used;
- exact search strings or major term groups;
- date of the search;
- inclusion criteria;
- exclusion criteria; and
- how references were screened.

The search does not need to be a systematic review, but another reader should understand how the five sources were chosen.

## 2. Complete the comparison table

For each source, record:

- citation and source type;
- task, intended population, and setting;
- dataset, sample size, observation unit, and split;
- input representation;
- architecture or method;
- activation and output head when relevant;
- training objective and optimizer when reported;
- baseline and controlled difference;
- metric definition and reported result;
- compute or resource information when available;
- strongest relevant contribution;
- material limitation; and
- relevance to the proposed project.

Use “not reported” for missing facts. Do not infer an implementation detail from an architecture label alone.

## 3. Classify comparability

For every important result pair, classify the comparison as:

- **directly comparable:** task, data, split, target, metric definition, and evaluation protocol align closely;
- **partially comparable:** some conditions align, but a named difference limits the conclusion; or
- **not directly comparable:** major differences prevent a meaningful numerical ranking.

Explain the classification. A higher score from a different dataset, test population, split, metric, or tuning policy does not prove a better method.

## 4. Identify the baseline

Name the simplest credible baseline for the project and explain:

- which sources establish its relevance;
- whether implementation details are available;
- whether it fits the proposed data and output task;
- which metric and split should be reproduced; and
- what result would make a more complex model worth testing.

The baseline must be feasible within the semester and should be implemented before a deep architecture is required.

## 5. Write the synthesis

Organize the synthesis around claims rather than one paragraph per paper. Address:

1. what the strongest sources agree on;
2. where methods, data, or evaluation differ;
3. which numerical results are directly comparable and which are not;
4. the simplest credible baseline;
5. the most important limitation in existing evidence;
6. one unresolved question the available project data can address; and
7. the consequence for the project's first controlled experiment.

Distinguish the authors' reported findings from your interpretation.

## Assessment rubric: 30 points

| Criterion | Points | Full-credit evidence |
| --- | ---: | --- |
| Search scope and selection | 4 | Reproducible search description with justified inclusion and exclusion |
| Source quality and relevance | 5 | At least five credible sources centered on primary evidence |
| Faithful evidence extraction | 6 | Accurate task, data, method, baseline, metric, and result with missing details marked |
| Comparability analysis | 5 | Direct, partial, and invalid comparisons are distinguished through protocol conditions |
| Baseline decision | 4 | Feasible baseline is supported by literature and matched to the proposed task |
| Synthesis and research gap | 4 | Claims integrate sources and lead to a testable unresolved question |
| Citation and presentation | 2 | Traceable citations, readable table, and clear distinction between fact and interpretation |

## Submission check

- [ ] The research question and search date are stated.
- [ ] Search terms, sources, inclusion, and exclusion criteria are recorded.
- [ ] At least five credible sources are compared.
- [ ] Primary papers support the main technical claims.
- [ ] Dataset, split, target, metric, baseline, and architecture are extracted where available.
- [ ] Missing details are marked “not reported.”
- [ ] Every major score comparison is classified and justified.
- [ ] The simplest credible baseline is feasible this semester.
- [ ] The synthesis is organized around evidence and disagreement.
- [ ] One unresolved, testable question is identified.
- [ ] Citations include persistent links or DOIs where available.
