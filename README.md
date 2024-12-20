# Stellar Classification - README

## Introduction

This project involves the analysis and classification of stellar objects using the **[Stellar Classification Dataset (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)**. The dataset is derived from the Sloan Digital Sky Survey (SDSS) and includes features that describe various properties of celestial objects, enabling their classification into stars, galaxies, or quasars.

## Dataset Description

The dataset contains the following attributes:

- **obj_ID**: Unique identifier for each object in the image catalog.
- **alpha**: Right Ascension angle (J2000 epoch).
- **delta**: Declination angle (J2000 epoch).
- **u**: Ultraviolet filter magnitude in the photometric system.
- **g**: Green filter magnitude in the photometric system.
- **r**: Red filter magnitude in the photometric system.
- **i**: Near-infrared filter magnitude in the photometric system.
- **z**: Infrared filter magnitude in the photometric system.
- **run_ID**: Run number used to identify the specific scan.
- **rereun_ID**: Rerun number specifying how the image was processed.
- **cam_col**: Camera column identifying the scanline within the run.
- **field_ID**: Field number identifying each field.
- **spec_obj_ID**: Unique ID for optical spectroscopic objects.
- **class**: Object class ("Galaxy", "Star", "Quasar").
- **redshift**: Redshift value based on the increase in wavelength.
- **plate**: Plate ID identifying each plate in SDSS.
- **MJD**: Modified Julian Date, indicating when the data was collected.
- **fiber_ID**: Fiber ID identifying the light focal plane in each observation.

**[source](https://www.kaggle.com/code/gsabhinav/multi-class-classification-noteboook)**

## Implementation

### 1. Data Loading and Exploration

The dataset is loaded using Pandas, and initial exploration includes:

- Displaying the first few rows.
- Checking data types and non-null values.
- Identifying and handling missing values.

### 2. Data Preprocessing

- **Feature Scaling**: StandardScaler is applied to normalize the data.
- **Train-Test Split**: Data is split into training (80%) and testing (20%) sets.

### 3. Classification Models

Several classification algorithms are implemented and evaluated:

#### K-Nearest Neighbors (KNN)

- Parameters: `n_neighbors=5`.
- Evaluates accuracy and generates a classification report.

#### Decision Tree Classifier

- Parameters: `random_state=42`.
- Trains and evaluates using accuracy and a classification report.

#### Naive Bayes (GaussianNB)

- Evaluates performance using accuracy and classification metrics.

#### Support Vector Machine (SVM)

- Parameters: `kernel='linear'` (modifiable).
- Trains and evaluates the model using accuracy and classification metrics.

#### Neural Network (MLPClassifier)

- Parameters: `hidden_layer_sizes=(100, 50)`, `max_iter=500`, `random_state=42`.
- Trains the network and evaluates using accuracy and a classification report.

### 4. Evaluation

- **Confusion Matrix**: Displays performance for the best model.
- **Classification Reports**: Summarize precision, recall, and F1-score for all models.

## Results

The project evaluates and compares the performance of multiple classifiers to identify the most suitable model for stellar classification based on accuracy and detailed metrics.

## Prerequisites

- Python (>= 3.7)
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

## How to Run

1. Clone this repository.
2. Ensure the dataset (`star_classification.csv`) is located in the appropriate directory.
3. Install the required libraries using `pip install -r requirements.txt`.
4. Run the script:

   ```bash
   python stellar_classification.py
   ```

## Future Work

- Implement other classifications
- Implement helper and classification choice on run
- GUI

## Acknowledgments

- **SDSS** for providing the dataset.
- Scikit-learn for machine learning tools.

---

This project is a step toward understanding and classifying celestial objects through computational techniques. Contributions and suggestions are welcome!
