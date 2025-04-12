# Investigating the effect of Imbalanced data on Trusthworthyness and Explainability with the UTKFace dataset.
Final project for Trustworthy &amp; Explainable AI

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [File Structure](#file-structure)  
4. [Setup and Environment](#setup-and-environment)  
5. [Usage](#usage)  
6. [References](#references)
7. [LLM Usage](#llm-usage)

---

## Project Overview
This project investigates how an **imbalanced dataset** affects the **trustworthiness** and **explainability** of a deep learning model. We use the **UTKFace dataset**, an unbalanced dataset containing over 20,000 face images labeled by age, gender, and ethnicity, to analyze the impact of imbalance on model performance and interpretability. We focus on classifying the age group of a person from facial images using deep learning. The model is trained to predict discrete age classes, each representing a 10-year age range (e.g., 20–29, 30–39). To enhance explainability, we:
- Provide the top 3 predicted age classes along with their associated confidence scores.
- Experiment with combining multiple predictions to form broader age ranges, aiming to reach at least 90% accuracy.
- Generate heatmaps that visually highlight the facial regions most influential in the model’s predictions.

This helps evaluate not only the accuracy of predictions, but also the trustworthiness and fairness of the model across different demographic groups.

### Key Objectives
- **Evaluate model performance** on imbalanced versus class weighting imbalanced data.
- **Assess explainability** using techniques like heatmaps and prototype/criticism analysis.
- **Identify bias and trust issues** in the model’s predictions due to dataset imbalance.

---

## Dataset
The [UTKFace dataset](https://susanqq.github.io/UTKFace/) contains face images that represent a broad spectrum of ages, genders, and ethnicities. However, the dataset is naturally imbalanced, which makes it an ideal candidate for exploring:
- How imbalance affects model training.
- The trustworthiness of predictions on underrepresented groups.
- The validity of explainability methods when data distributions are skewed.

---

## File Structure

```plaintext
.
├── datasets
│   └── utkface
│       ├── 1_0_0_20161219204759412.jpg.chip.jpg
│       ├── ... (other UTKFace images)
├── images
│   ├── prototypes_and_criticisms
│   ├── age_by_gender_and_ethnicity.png
│   ├── confusion_matrix_balanced.png
│   ├── confusion_matrix_base.png
│   ├── heatmap_output.jpg
│   └── (other visualization outputs)
├── models
│   ├── base_model_trained.h5
│   ├── best_hyperparameters_balanced.pkl
│   ├── best_hyperparameters.h5
│   ├── class_weight_model_trained.h5
├── hyperparameter_tuning
│   └── (keras-tuner files)
│── balanced_model_prototypes_and_criticisms.csv
│── base_model_prototypes_and_criticisms.csv
├── data_analysis.ipynb
├── training.ipynb
├── requirements.txt
├── utils.py
└── README.md
```

**Folder / File Descriptions:**
- **datasets/utkface**: Contains all UTKFace images organized as provided in the original dataset.
- **images/prototypes_and_criticisms**: Visualization outputs including prototypes, criticisms and heatmaps.
- **models**: Contains trained models (H5) and best hyperparameters.
- **hyperparameter_tuning**: Keras-tuner output.
- **data_analysis.ipynb**: Notebook for analyzing the distribution of the dataset (age, gender, ethnicity) and the impact of imbalance.
- **training.ipynb**: Main training notebook showing data loading and model training.
- **evaluation.ipynb**: Main evaluation notebook measuring performance of the models on the test set and their interpretability.
- **requirements.txt**: Lists the Python dependencies for the project.
- **utils.py**: Common helper functions.

---

## Setup and Environment

### Creating a Conda Environment
It is recommended to use a Conda environment to manage dependencies. Follow these steps:

1. **Create the Conda environment:**
   ```bash
   conda create -n utkface_env python=3.11
   ```

2. **Activate the environment:**
   ```bash
   conda activate utkface_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. **Prepare the Dataset:**
   - Place the UTKFace images in the `datasets/utkface` directory. Make sure the file naming is consistent with the UTKFace dataset conventions.

2. **Data Analysis:**
   - Run the `data_analysis.ipynb` notebook to view distribution plots and analyze dataset imbalance.

3. **Model Training:**
   - Execute the `training.ipynb` notebook to train the base model on the imbalanced dataset.

4. **Explainability Analysis:**
   - Use the trained models in `evaluation.ipynb` to measure their performance on the test set.
   - Generate visualizations and review prototypes and criticisms.
   - Evaluate heatmaps and other explainability metrics to determine model trustworthiness.

---

## References
- **UTKFace Dataset:** [Official UTKFace Page](https://susanqq.github.io/UTKFace/)
- **Kaggle UTKFace:** [Kaggle UTKFace Code Page](https://www.kaggle.com/datasets/jangedoo/utkface-new/code)
- **Tensorflow GradCam++ Implementation:** [tf-keras-vis GitHub Page](https://github.com/keisen/tf-keras-vis)
---


## LLM Usage

Large Language Models (LLMs) were used to assist in several small parts of this project, including:

- Assisting with code to generate graphs.
- Assisting with combining the heatmaps and their predicted outputs into one image.
- Minor assistance with errors in code.
- Writing a first (with many errors) version of the README.
