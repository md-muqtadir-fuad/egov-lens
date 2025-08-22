import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def data_overview(data):
    """
    Generate a basic overview of the data (columns, data types, etc.)
    """
    print("\n=== Data Overview ===")
    print(data.info())  # Basic structure
    print("\n=== First 5 rows of the data ===")
    print(data.head())

def missing_values_report(data):
    """
    Generate a report on missing values.
    """
    print("\n=== Missing Values Report ===")
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data / len(data)) * 100
    missing_report = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage': missing_percentage
    })
    print(missing_report[missing_report['Missing Values'] > 0].sort_values('Percentage', ascending=False))

    # Visualize missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Data Heatmap')
    plt.show()

def duplicate_report(data):
    """
    Check for duplicate rows in the dataset.
    """
    print("\n=== Duplicate Records Report ===")
    duplicate_rows = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")
    
    if duplicate_rows > 0:
        data = data.drop_duplicates()
        print(f"Removed {duplicate_rows} duplicate rows.")
    else:
        print("No duplicate rows found.")
    return data

def data_types_report(data):
    """
    Check the data types of the columns and suggest corrections.
    """
    print("\n=== Data Types Report ===")
    data_types = data.dtypes
    print(data_types)

    # Checking for columns that should have different types (example)
    # You can add your own conditions depending on the expected data types.
    for col in data.columns:
        if data[col].dtype == 'object':
            unique_values = data[col].nunique()
            if unique_values < 10:  # Possibly categorical
                print(f"Column '{col}' seems to be categorical with {unique_values} unique values.")
            else:
                print(f"Column '{col}' seems to be textual.")
    
def descriptive_statistics(data):
    """
    Generate descriptive statistics for numerical columns.
    """
    print("\n=== Descriptive Statistics ===")
    print(data.describe())

def correlation_heatmap(data):
    """
    Plot a correlation heatmap for numeric features.
    """
    print("\n=== Correlation Heatmap ===")
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

def visualize_categorical_features(data):
    """
    Plot the distribution of categorical features.
    """
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data[col])
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()

def visualize_numerical_features(data):
    """
    Plot the distribution of numerical features.
    """
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

def data_quality_report(file_path):
    """
    Generate a data quality report for the dataset.
    """
    # Load data
    data = load_data(file_path)
    if data is None:
        return

    # Generate reports
    data_overview(data)
    missing_values_report(data)
    data = duplicate_report(data)
    data_types_report(data)
    descriptive_statistics(data)
    correlation_heatmap(data)
    visualize_categorical_features(data)
    visualize_numerical_features(data)

if __name__ == "__main__":
    # Provide the file path of your CSV dataset here
    file_path = 'dataset.csv'  # Replace with your actual dataset path
    data_quality_report(file_path)
