# Heart Failure Prediction using Machine Learning

A comprehensive machine learning project for predicting heart failure using multiple classification algorithms. This project compares the performance of K-Nearest Neighbors (KNN), Naive Bayes, Decision Tree, and Random Forest classifiers.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Algorithms Used](#algorithms-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)

## 🔍 Overview

This project implements and compares four different machine learning classifiers to predict heart failure based on various medical features. The dataset contains patient information including age, blood pressure, cholesterol levels, and other cardiovascular indicators.

## 📊 Dataset

The dataset (`heart_failure.csv`) contains the following features:

- **age**: Age of the patient
- **sex**: Gender of the patient
- **cp**: Chest pain type
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol
- **fbs**: Fasting blood sugar
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **target**: The target variable (presence or absence of heart disease)

## 🎯 Features

- **Data Preprocessing**: Handles missing values and outliers using IQR method
- **Feature Encoding**: Converts categorical variables to numerical values
- **Multiple Classifiers**: Implements KNN, Naive Bayes, Decision Tree, and Random Forest
- **Performance Metrics**: Calculates accuracy, precision, recall, and confusion matrices
- **Visualizations**: 
  - Decision tree visualization
  - Feature importance chart

## 🤖 Algorithms Used

1. **K-Nearest Neighbors (KNN)**: K value calculated as √(training set size)
2. **Naive Bayes**: Gaussian Naive Bayes classifier
3. **Decision Tree**: Standard decision tree classifier
4. **Random Forest**: Ensemble method with 100 estimators, max_depth=25

## 📦 Installation

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib graphviz
```

### Additional Requirements

For graphviz visualization, you may need to install Graphviz on your system:
- **Windows**: Download from [Graphviz website](https://graphviz.org/download/)
- **Linux**: `sudo apt-get install graphviz`
- **Mac**: `brew install graphviz`

## 🚀 Usage

### Running the Main Script

```bash
python heart_failure.py
```

The script will:
1. Load and preprocess the data
2. Remove outliers from specified columns (age, thalach, trestbps)
3. Split data into training (70%) and testing (30%) sets
4. Train and evaluate all four classifiers
5. Generate performance reports and visualizations

## 📈 Results

### Model Performance Comparison

The script outputs detailed performance metrics for each classifier including:
- Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report

### Feature Importance

The Decision Tree model identifies the most important features for prediction:

![Feature Importance](Figure_1.png)

### Code Execution Results

![Execution Results](image.png)

## 📁 Project Structure

```
heart_failure/
│
├── heart_failure.py          # Main script with all classifiers
├── heart_failure.csv          # Dataset
├── datasetvariables.txt       # Feature descriptions
├── README.md                  # This file
├── .gitignore                 # Git ignore file
├── Figure_1.png              # Feature importance visualization
├── image.png                 # Code execution results
├── Tree.png                  # Decision tree visualization
└── Tree.svg                  # Decision tree (SVG format)
```

## 🔧 Configuration

You can modify the following parameters in `heart_failure.py`:

- **Test size**: Change `test_size` in `split_data()` function (default: 0.3)
- **Random state**: Modify `random_state` for reproducibility (default: 13)
- **Outlier columns**: Update `columns_to_check` list to change which columns are checked for outliers
- **Random Forest parameters**: Adjust `n_estimators`, `max_depth`, etc. in `apply_random_forest_classifier()`

## 📝 Notes

- The script uses median imputation for missing values
- Outliers are removed using the Interquartile Range (IQR) method
- All categorical features are label-encoded
- The decision tree visualization is saved as `Tree.png`

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Heart Failure Prediction Project

---

**Note**: This project is for educational and research purposes. Always consult medical professionals for actual health-related decisions.
