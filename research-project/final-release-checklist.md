# Final project release checklist

Complete this checklist before recording the final commit or archive hash.

## Repository identity

- [ ] repository URL or permitted archive location is recorded;
- [ ] exact commit, release, or archive SHA-256 is recorded;
- [ ] working state is clean or all deviations are documented;
- [ ] README states the research question and strongest supported claim;
- [ ] repository contains no unrelated caches, downloads, or temporary outputs.

## Environment and execution

- [ ] supported Python version and operating system are stated;
- [ ] environment-construction command is tested;
- [ ] resolved package versions are preserved;
- [ ] required hardware, memory, storage, and expected runtime are stated;
- [ ] one command regenerates one central result from the repository root;
- [ ] expected output path and agreement criterion are stated;
- [ ] known nondeterminism is documented.

## Data

- [ ] source, version, access date or identifier, and license are recorded;
- [ ] unit of observation, target, population, and exclusions are defined;
- [ ] preprocessing fitted on training data is documented;
- [ ] train, validation, and locked-test construction is reproducible;
- [ ] restricted data have a permitted access route or synthetic audit path;
- [ ] private or identifying records are excluded from the release.

## Experiment

- [ ] baseline and controlled factor are explicit;
- [ ] configuration actually used is preserved;
- [ ] seed set and source of uncertainty are named;
- [ ] model-selection rule and number of candidates are recorded;
- [ ] test-data use is disclosed;
- [ ] result files identify validation, test, and exploratory evidence.

## Report and presentation

- [ ] title, abstract, main result, conclusion, and talk agree;
- [ ] every central table and figure has a source or generating command;
- [ ] published and project-generated results are distinct;
- [ ] uncertainty and sample counts are defined;
- [ ] errors, subgroup results, or robustness evidence are included where relevant;
- [ ] limitations identify affected claims and likely consequences;
- [ ] report and presentation source files are included;
- [ ] PDF exports open correctly;
- [ ] slides are readable, accessible, and timed.

## Reproduction and integrity

- [ ] Week 13 first attempt, correction, and verification are linked;
- [ ] central artifact hashes match the release manifest;
- [ ] static Week 14 release audit passes;
- [ ] generated outputs required for review are identified;
- [ ] a failed or partial reproduction remains documented honestly.

## License, citation, and responsible release

- [ ] project code and text licenses are compatible with included material;
- [ ] datasets, pretrained models, external code, and figures are attributed;
- [ ] citation information is provided;
- [ ] automated or external assistance is disclosed under the course policy;
- [ ] privacy, consent, ethics, security, and misuse concerns are addressed;
- [ ] credentials, tokens, private keys, and personal paths are absent;
- [ ] unavailable or omitted artifacts are named with reasons.

## Research direction

- [ ] one observed boundary motivates the direction;
- [ ] two competing explanations are stated;
- [ ] the next experiment distinguishes them;
- [ ] metric, uncertainty unit, decision threshold, and stop condition are defined;
- [ ] data, compute, skills, collaborators, and approvals are assessed;
- [ ] current stage is labeled as feasibility, master's, PhD, or applied research;
- [ ] the direction remains useful under a negative result.

## Submission

- [ ] correct institutional channel and filename rules are followed;
- [ ] access permissions were tested from another account or reviewer role;
- [ ] final submission record contains all artifact identifiers;
- [ ] author retained a permitted backup;
- [ ] student work is not made public without the required permissions.
