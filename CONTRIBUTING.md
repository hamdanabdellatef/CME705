# Contributing

Contributions should improve a specific learning outcome, correct an error, or make an experiment more reproducible.

## Before proposing a change

- Open an issue describing the teaching problem and intended learner.
- Keep examples small enough to run on a CPU unless the activity explicitly requires an accelerator.
- Add provenance and license information for every external asset or adapted fragment.
- Use generated or openly licensed data in examples.
- Never include student records, private data, credentials, or unpublished student work.

## Validate a change

```bash
python -m compileall labs tests
python -m pytest
```

For a new experiment, state the expected learning result, seed, runtime, output shape, metric, and limits of the demonstration. Tests should verify a meaningful mathematical or interface property.
