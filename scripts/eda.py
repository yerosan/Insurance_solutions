# eda.py - Exploratory Data Analysis (EDA) for Insurance Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class EDA:
    def __init__(self, data_path):
        """
        Initializes the EDA class with the path to the dataset.
        """
        self.data_path = data_path

    def read_data(self):
        """
        Reads the dataset from the specified path.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        df = pd.read_csv(self.data_path, delimiter="|",low_memory=False)
        return df

    def summarize_data(self, df):
        """
        Prints summary statistics and data types of the dataset.

        Args:
            df (pd.DataFrame): The dataset to summarize.
        """
        print("---- Descriptive Statistics ----")
        print(df.describe())
        print("\n---- Data Types ----")
        print(df.dtypes)

    def assess_data_quality(self, df):
        """
        Assesses data quality by checking for missing values and duplicates.

        Args:
            df (pd.DataFrame): The dataset to assess.
        """
        print("\n---- Missing Values ----")
        print(df.isnull().sum())
        print("\n---- Duplicates ----")
        print(f"Number of duplicate rows: {df.duplicated().sum()}")


    def clean_data(self, df):
        """
        Cleans the dataset by handling missing values in both categorical and numerical columns.

        Args:
            df (pd.DataFrame): The dataset to clean.

        Returns:
            pd.DataFrame: The cleaned dataset with missing values handled.
        """
        
        # Set the threshold for dropping columns with too many missing values (less than 60% non-null values)
        threshold = len(df) * 0.6
        
        # Drop columns with missing values exceeding the threshold
        df_cleaned = df.dropna(thresh=threshold, axis=1)
        
        # Loop through each column to handle missing values
        for col in df_cleaned.columns:
            if col == df_cleaned.index.name:  # Skip the index column
              continue
            if df_cleaned[col].dtype == "object":
                # Fill missing values in categorical columns with the most frequent value (mode)
                df_cleaned.loc[:, col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            else:
                # Fill missing values in numerical columns with the mean value
                df_cleaned.loc[:, col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            
            # Remove duplicate rows from the DataFrame
        df_cleaned = df_cleaned.drop_duplicates()
        
        return df_cleaned


    def plot_histograms(self, df):
        """
        Plots histograms for numerical columns to visualize their distributions.

        Args:
            df (pd.DataFrame): The dataset containing the numerical columns.
        """
        columns = [
            "TotalPremium", "TotalClaims", "SumInsured", "CalculatedPremiumPerTerm", 
            "ExcessSelected", "VehicleType", "make", "Model", "RegistrationYear", 
            "cubiccapacity", "kilowatts", "IsVATRegistered", 
            "MaritalStatus", "Gender", "Province", "PostalCode", "MainCrestaZone", 
            "SubCrestaZone", "CoverType", "CoverCategory", "CoverGroup", "Product"
        ]

        df = df[columns]
        numeric_df=df.select_dtypes(include=['number'])
        num_columns = numeric_df.columns
        for col in num_columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def plot_bar_charts(self, df):

       
        """
        Plots bar charts for categorical columns to visualize the frequency of top values.

        Args:
            df (pd.DataFrame): The dataset containing the categorical columns.
        """

        columns = [
            "TotalPremium", "TotalClaims", "SumInsured", "CalculatedPremiumPerTerm", 
            "ExcessSelected", "VehicleType", "make", "Model", "RegistrationYear", 
            "cubiccapacity", "kilowatts", "IsVATRegistered", 
            "MaritalStatus", "Gender", "Province", "PostalCode", "MainCrestaZone", 
            "SubCrestaZone", "CoverType", "CoverCategory", "CoverGroup", "Product"
        ]

        df = df[columns]
        cat_df=df.select_dtypes(include=['object'])
        cat_columns = cat_df.columns
        for col in cat_columns:
            # Get the top 15 most frequent values
            top_15_values = df[col].value_counts().nlargest(15).index

            # Filter the DataFrame to include only the top 15 values
            df_filtered = df[df[col].isin(top_15_values)]
            plt.figure(figsize=(10, 5))

            sns.countplot(data=df_filtered, x=col)
            plt.xticks(rotation=45)
            plt.title(f"Bar Chart for {col}")
            plt.show()

    def plot_scatter(self, df, x_col, y_col):
        """
        Plots a scatter plot to examine the relationship between two numerical variables.

        Args:
            df (pd.DataFrame): The dataset containing the variables.
            x_col (str): The name of the column to plot on the x-axis.
            y_col (str): The name of the column to plot on the y-axis.
        """
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f"Scatter Plot of {x_col} vs {y_col}")
        plt.show()

    def correlation_matrix(self, df):
        """
        Plots a heatmap of the correlation matrix for selected numerical variables.

        Args:
            df (pd.DataFrame): The dataset containing the numerical variables.
        """
        plt.figure(figsize=(12, 8))
        corr = df[['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

    def detect_outliers(self, df):
        """
        Plots box plots to detect outliers in numerical columns.

        Args:
            df (pd.DataFrame): The dataset containing the numerical columns.
        """
        num_columns = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
        for col in num_columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(df[col])
            plt.title(f"Box Plot for {col}")
            plt.show()

    def plot_creative(self, df):
        """
        Creates and displays creative visualizations for key insights.

        Args:
            df (pd.DataFrame): The dataset to visualize.
        """
        # Total Premium vs Total Claims by Province
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Province', y='TotalPremium', data=df)
        plt.title("Total Premium Distribution by Province")
        plt.xticks(rotation=45)
        plt.show()

        # Claims vs Premium with regression
        plt.figure(figsize=(10, 6))
        sns.regplot(x='TotalPremium', y='TotalClaims', data=df, scatter_kws={'s':10}, line_kws={'color':'red'})
        plt.title("Regression between TotalPremium and TotalClaims")
        plt.show()

  