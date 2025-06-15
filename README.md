# Speaker Gender and Age Recognition

This project predicts the **gender** and **age group** of a speaker from an audio recording using advanced machine learning techniques. The repository provides a complete pipeline from audio preprocessing and feature engineering to model training, evaluation, and deployment.

---

## Table of Contents

- [Overview](#overview)
- [Technologies & Libraries](#technologies--libraries)
- [Feature Engineering](#feature-engineering)
- [Audio Preprocessing](#audio-preprocessing)
- [Modeling Approach](#modeling-approach)
- [Feature Selection](#feature-selection)
- [Parallelization & Performance](#parallelization--performance)
- [Usage](#usage)
- [References](#references)
- [Research Papers](#research-papers)
- [Deployment (Docker)](#deployment-docker)

---

## Overview

This repository contains scripts and resources to train and evaluate models for speaker gender and age recognition from audio files. The workflow includes robust feature extraction, model stacking, and reproducible evaluation.

---

## Technologies & Libraries

- **Programming Language:** Python 3.x
- **Core Libraries:**  
  - `librosa` (audio processing)  
  - `parselmouth` (pitch/formant extraction)  
  - `numpy`, `scipy`, `pandas` (data handling, statistics)  
  - `scikit-learn, tensorflow` (machine learning, feature selection, stacking)  
  - `pydub` (audio manipulation)
- **Other Tools:**  
  - `multiprocessing` (parallel feature extraction)  
  - `docker` (containerization)

---

## Feature Engineering

- **Engineered Features:**  
  - MFCCs (mean, median, std)
  - STFT statistics (mean, std, median)
  - Spectral features (centroid, bandwidth, contrast, flatness, rolloff, zero-crossing rate)
  - Pitch and formant features (mean, std, range, contour)
  - Rhythm and intonation metrics (pause duration, speaking rate, etc.)
- **Raw Features:**  
  - Directly extracted from audio without transformation (e.g., raw pitch contour)

---

## Audio Preprocessing

- **Silence Removal:** Focuses on speech segments by removing silence.
- **Loudness Normalization:** Standardizes volume across samples.
- **Pause & Phoneme Detection:** Identifies speech and silence intervals.
- **Feature Extraction:** Automated, parallelized extraction of all features.

---

## Modeling Approach

- **Models Evaluated:**  
  - Random Forest, SVM, Gradient Boosting, and others.
- **Stacking Ensemble:**  
  - Final model uses a stacking approach, combining predictions from multiple base models for improved accuracy.
- **Postprocessing:**  
  - Final predictions may be smoothed or thresholded for stability.

---

## Feature Selection

- **Statistical Analysis:**  
  - Feature distributions visualized and analyzed.
- **ANOVA Testing:**  
  - Used to select features most relevant to age and gender prediction.

---

## Parallelization & Performance

- **Threading/Multiprocessing:**  
  - Feature extraction is parallelized to speed up processing of large datasets.

---

## Usage

### 1. Prepare Your Data

- Place your test audio files in the `test` directory (or replace with your own test cases).

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run with Docker (Recommended)

```bash
chmod +x docker_script.sh
sh docker_script.sh 1
```

### 4. Run Locally (Alternative)

- Extract features:
  ```bash
  python scripts/extract_features.py --input test/
  ```
- Predict:
  ```bash
  python scripts/predict.py --input test/
  ```

### 5. Scripts

- `extract_features.py`: Feature extraction  
- `train.py`: Model training  
- `predict.py`: Inference on new audio

---

## References

- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Praat & Parselmouth](https://parselmouth.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- Academic papers and resources used for feature engineering and modeling (list here as needed).

---

## Research Papers

Below are some of the key research papers and resources that inspired or guided this project:

- [1] Schuller, B., et al. "Automatic Recognition of Emotions and Personality Traits from Speech." IEEE Signal Processing Magazine, 2013.
- [2] Li, Haizhou, et al. "Speaker Age Estimation Using i-vectors." INTERSPEECH, 2013.
- [3] Bahari, M., et al. "Speaker Age Estimation Using Hidden Markov Model Weight Supervectors." Interspeech, 2012.
- [4] Sahidullah, Md, et al. "Design, Analysis and Experimental Evaluation of Block Based Transformation in MFCC Computation for Speaker Recognition." Speech Communication, 2015.
- [5] Other relevant papers and resources as used in the project.

---

## Deployment (Docker)

The project includes a Dockerfile for easy deployment.

**Build and run:**
```bash
docker build -t speaker-age-gender .
docker run -v $(pwd)/test:/app/test speaker-age-gender
```

---

## Acknowledgements

- Thanks to the authors of the open-source libraries and papers that inspired this work.

---
