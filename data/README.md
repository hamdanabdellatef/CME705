# Data policy

Datasets are not stored in this first release. Every lab or project dataset must provide:

- an official source and version or retrieval date;
- license and access conditions;
- a checksum when redistribution or stable download permits it;
- a data dictionary and unit of observation;
- a deterministic split-generation script or official split description;
- a statement about sensitive attributes, privacy, consent, and intended use; and
- instructions that fit inside a fresh environment.

Fit imputation, scaling, feature selection, augmentation policies, and other learned preprocessing using the training partition only. Keep the final test partition unavailable for tuning. Use group-aware or time-aware splitting when observations are dependent.

Local data paths `data/raw/` and `data/processed/` are ignored by Git.
