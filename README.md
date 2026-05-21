# Kaggle - Deep Past Initiative Machine Translation

Public inference script used for my `Deep Past Initiative Machine Translation` submissions on Kaggle.

Best visible private leaderboard score from this public submission workflow: `33.2808`.

## Files

- `solution.py`: Kaggle inference script

## Origin

This public Kaggle notebook is a fork of:

- `manwithacat/byt5-improved-v1-0-0-host-guidance`
- Author: `manwithacat`

Important context:

- The Kaggle notebook itself is a forked inference notebook.
- The data preparation, model training, and experimentation behind this run were done on my side in Colab.
- The released Kaggle script is the public submission/inference layer, not the full Colab training pipeline.

## My modifications

Compared with the upstream public notebook/fork lineage, this version is documented as:

- Host-guidance preprocessing for transliteration normalization
- ASCII to diacritic normalization
- Gap marker normalization
- Optional consonantal skeleton preprocessing
- Repetition penalty and decoding tweaks
- Wiring to the model artifact used for submission

## Notes

- This repository preserves the public Kaggle submission script as published.
- If I later export the training pipeline from Colab, it should live in a separate training-focused repository.
