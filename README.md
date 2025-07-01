# BeatPatrol - Emotion-Aware Genre Classification with Deep Audio Representations

## Overview

BeatPatrol is a deep learning pipeline for music genre classification and emotion regression using spectrogram-based audio representations. It leverages CNNs, LSTMs, and Audio Spectrogram Transformers (AST) to learn from mel-spectrograms, chromagrams, and beat-synced features. The project explores multitask learning, transfer learning, and representation analysis to bridge low-level audio with high-level affective attributes.

## Features

* Genre classification across 10 curated music classes
* Emotion regression on valence, energy, and danceability
* Deep architectures: CNNs, LSTMs, and Audio Spectrogram Transformers
* Transfer and multitask learning support
* Beat-synchronization and class remapping in preprocessing
* Robust evaluation with classification and regression metrics
* Visual analysis of learned audio representations

## Datasets

* **FMA Subset**: Free Music Archive dataset trimmed to 10 balanced genres for classification
* **Emotion Regression Dataset**: Annotated with valence, energy, and danceability values per track

## Evaluation Metrics

* **Classification**: Accuracy, Precision, Recall, Macro-F1
* **Regression**: Mean Squared Error (MSE), Pearson & Spearman Correlation

## How to Use

1. Run preprocessing notebooks (`step_1_2_3.ipynb`) to generate spectrograms
2. Train models via the sequential notebooks (`step_4_5_6.ipynb` through `step_11.ipynb`)
3. Load `model_weights/` and run evaluation and analysis

## Authors

- [Evangelos Chaniadakis](https://github.com/xaniadakis)
- [Alexios Filippakopoulos](https://github.com/alexisfilippakopoulos)

---
