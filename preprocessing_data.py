import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the modified data
file_path = 'heart_failure.csv'
data = pd.read_csv(file_path)

# Display basic information and statistics
print(data.info())
print(data.describe().round(2))

# Preprocessing Data
num_unique_age = data['age'].nunique()
unique_age = data['age'].unique()
age_value_counts = data['age'].value_counts()

num_unique_sex = data['sex'].nunique()
unique_sex = data['sex'].unique()
sex_value_counts = data['sex'].value_counts()

num_unique_cp = data['cp'].nunique()
unique_cp = data['cp'].unique()
cp_value_counts = data['cp'].value_counts()

# Check for missing values and duplicates
missing_values = data.isna().sum()
num_duplicates = data.duplicated().sum()

# Clean the data: Remove duplicates and handle missing values
print(f"Data shape before cleaning: {data.shape}")
print(f"Missing values:\n{missing_values}")
print(f"Number of duplicate rows: {num_duplicates}")

# Remove duplicate rows
if num_duplicates > 0:
    data = data.drop_duplicates()
    print(f"Removed {num_duplicates} duplicate rows")

# Handle missing values - fill numeric columns with median, categorical with mode
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(include=['object']).columns

for col in numeric_columns:
    if data[col].isna().sum() > 0:
        median_value = data[col].median()
        data[col].fillna(median_value, inplace=True)
        print(f"Filled missing values in {col} with median: {median_value}")

for col in categorical_columns:
    if data[col].isna().sum() > 0:
        mode_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
        data[col].fillna(mode_value, inplace=True)
        print(f"Filled missing values in {col} with mode: {mode_value}")

print(f"Data shape after cleaning: {data.shape}")

# Sample data for exploration
print(data.sample(10))

# Exploratory Data Analysis
data['age'].value_counts().plot.bar()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Adding additional columns

# Age Group column
age_bins = [39, 50, 60, 70, 80, 90]
age_labels = ['40-50', '51-60', '61-70', '71-80', '81-90']
data['Age Group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels)

# Blood Pressure Category column
def bp_category(trestbps):
    if trestbps < 120:
        return 'Normal'
    elif 120 <= trestbps < 130:
        return 'Elevated'
    elif 130 <= trestbps < 140:
        return 'Hypertension Stage 1'
    else:
        return 'Hypertension Stage 2'

data['Blood Pressure Category'] = data['trestbps'].apply(bp_category)

# Cholesterol Category column
def chol_category(chol):
    if chol < 200:
        return 'Desirable'
    elif 200 <= chol < 240:
        return 'Borderline High'
    else:
        return 'High'

data['Cholesterol Category'] = data['chol'].apply(chol_category)

# Heart Rate Zone column
def hr_zone(thalach):
    if thalach < 100:
        return 'Below Average'
    elif 100 <= thalach < 140:
        return 'Average'
    else:
        return 'Above Average'

data['Heart Rate Zone'] = data['thalach'].apply(hr_zone)

# Risk Level column based on multiple factors
def risk_level(row):
    risk = 0
    if row['Blood Pressure Category'] in ['Hypertension Stage 1', 'Hypertension Stage 2']:
        risk += 1
    if row['Cholesterol Category'] == 'High':
        risk += 1
    if row['thalach'] < 100:
        risk += 1
    if row['age'] >= 60:
        risk += 1
    return 'High' if risk >= 3 else 'Moderate' if risk == 2 else 'Low'

data['Risk Level'] = data.apply(risk_level, axis=1)

# Save the modified data to a new CSV file
output_file_path = 'heart_failure_enhanced.csv'
data.to_csv(output_file_path, index=False)

# Print results for review
print(f"Number of unique ages: {num_unique_age}")
print(f"Unique ages: {unique_age}")
print(age_value_counts)

print(f"Number of unique sexes: {num_unique_sex}")
print(f"Unique sexes: {unique_sex}")
print(sex_value_counts)

print(f"Number of unique chest pain types: {num_unique_cp}")
print(f"Unique chest pain types: {unique_cp}")
print(cp_value_counts)

# Print final data quality metrics after cleaning
final_missing_values = data.isna().sum()
final_duplicates = data.duplicated().sum()
print(f"\nFinal data quality check:")
print(f"Missing values after cleaning:\n{final_missing_values}")
print(f"Number of duplicate rows after cleaning: {final_duplicates}")
print(f"Final data shape: {data.shape}")

print("\nFirst 5 rows of cleaned and enhanced data:")
print(data.head())
