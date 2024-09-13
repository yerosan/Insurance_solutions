# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


import pandas as pd

class EDA:
    def __init___(self,data_path):
        self.data_path=data_path
         
    def reading_data(self,data_path):
        df=pd.read_csv(data_path, delimiter="|")
        return df

    # 1. Data Summarization

    def summarize_data(self, df):
        print("---- Descriptive Statistics ----")
        print(df.describe())
        print("\n---- Data Types ----")
        print(df.dtypes)

        
    # 2. Data Quality Assessment

    def assess_data_quality(self,df):
        print("\n---- Missing Values ----")
        print(df.isnull().sum())
        print("\n---- Duplicates ----")
        print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # 3. Data Cleaning
    def clean_data(self,df):
        # Handling missing values for categorical columns
        df=pd.DataFrame(df)
        cat_cols = ['Bank', 'AccountType', 'MaritalStatus', 'Gender', 'VehicleType', 'make', 'Model',"mmcode"]
        for col in cat_cols:
            df[col]=df[col].fillna(df[col].mode()[0])  # Impute with mode

        # Handling missing values for numerical columns (impute with median or mean)
        num_cols = ['SumInsured', 'TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'CustomValueEstimate']
        for col in num_cols:
            df[col]=df[col].fillna(df[col].mean())  # Impute with median


        return df

    # 4. Univariate Analysis

    def plot_histograms(self,df):
        num_columns = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
       

        for col in num_columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def plot_bar_charts(self,df):
        cat_columns = ['Province', 'VehicleType', 'Gender', 'CoverCategory', 'MaritalStatus']
        for col in cat_columns:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x=col)
            plt.title(f"Bar Chart for {col}")
            plt.show()

    # 5. Bivariate/Multivariate Analysis

    def plot_scatter(self,df, x_col, y_col):
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f"Scatter plot of {x_col} vs {y_col}")
        plt.show()

    def correlation_matrix(self,df):
        plt.figure(figsize=(12, 8))
        corr = df[['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']].corr()
        # corr = df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

    # 6. Outlier Detection using Box Plots

    def detect_outliers(self,df):
        num_columns = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
        for col in num_columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(df[col])
            plt.title(f"Box Plot for {col}")
            plt.show()

    # 7. Visualization - Key Insights (Creative Plots)

    def plot_creative(self,df):
        # Example: Total Premium vs Total Claims by Province
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Province', y='TotalPremium', data=df)
        plt.title("Total Premium Distribution by Province")
        plt.xticks(rotation=45)
        plt.show()

        # Example: Claims vs Premium with regression
        plt.figure(figsize=(10, 6))
        sns.regplot(x='TotalPremium', y='TotalClaims', data=df, scatter_kws={'s':10}, line_kws={'color':'red'})
        plt.title("Regression between TotalPremium and TotalClaims")
        plt.show()

  