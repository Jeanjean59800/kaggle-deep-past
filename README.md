# Kaggle - Deep Past Initiative Machine Translation

This repo contains the public inference script I used for `Deep Past Initiative Machine Translation`.

Best visible private leaderboard score from this submission workflow: `33.2808`.

## File

- `solution.py`: Kaggle inference script

## Origin

The public Kaggle notebook I used here is a fork of:

- `manwithacat/byt5-improved-v1-0-0-host-guidance`
- author: `manwithacat`

What matters is that the Kaggle notebook is only the public inference/submission layer.

The data work, the training, and most of the experimentation for this run were done by me in Colab on my own side. So this repository is not the full training pipeline. It is the public script I used to run submissions on Kaggle.

## What I changed

- host-guidance preprocessing for transliteration normalization
- ASCII to diacritic normalization
- gap marker normalization
- optional consonantal skeleton preprocessing
- repetition penalty and decoding tweaks
- connection to the model artifact I used for submission

## Note

If I clean up the Colab training pipeline later, I will probably publish it separately.
