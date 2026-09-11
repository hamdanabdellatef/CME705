# Week 14 further reading — Research Communication and Release

Use these sources to revise the final presentation and release. The course materials summarize principles but do not redistribute external figures or checklists.

## Presentation design

[Naegle, “Ten simple rules for effective presentation slides”](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009554) argues for one main idea per slide, conclusion-bearing headings, essential content, effective graphics, iterative practice, and technical fallbacks.

Read for:

- turning a section label into a message;
- deciding when one figure is too complex;
- connecting slide design to speaking time;
- preparing a PDF or static fallback.

[Bourne, “Ten Simple Rules for Making Good Oral Presentations”](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.0030077) discusses audience, logical flow, practice, time, and effective visuals.

Read for planning for the audience, rehearsing delivery, and using questions as scientific discussion.

## Reporting and claims

The [NeurIPS Paper Checklist guidelines](https://neurips.cc/public/guides/PaperChecklist) cover claims, limitations, assumptions, reproducibility, experimental details, statistical evidence, compute, ethics, licenses, and released assets.

Use the checklist to audit whether the talk omits a limitation that changes the conclusion, whether uncertainty and compute are described, and whether assets and licenses are traceable.

## Artifact review

The [ACM Artifact Review and Badging policy](https://www.acm.org/publications/policies/artifact-review-and-badging-current) distinguishes artifact availability, artifact quality, and validation of results.

Use it to separate public availability, functional and documented packaging, and independent validation of a result. The Week 14 static audit checks package declarations and hashes. The Week 13 reproduction attempt provides execution evidence.

## Repository releases

[GitHub’s release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) explains how a release can bundle a tagged repository state, notes, and downloadable assets.

Before using a hosted release:

- verify the tag resolves to the reviewed commit;
- verify attached report and presentation match the submission;
- record whether release assets can later change;
- avoid attaching data or weights without redistribution rights.

A hosted release is optional. A fixed commit or permitted archive hash is sufficient when public hosting is inappropriate.

## Reporting artifacts

Review:

- [Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596);
- [Datasheets for Datasets](https://doi.org/10.1145/3458723);
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Use the concepts that apply to the project rather than copying a full template mechanically.

## Reading prompts

1. Which slide in your deck currently contains more than one main idea?
2. Which title makes a claim that the displayed evidence cannot verify?
3. Which report detail belongs in a backup slide rather than the main story?
4. Does the release identify availability, functionality, and result validation separately?
5. Which limitation produces the strongest next research question?
