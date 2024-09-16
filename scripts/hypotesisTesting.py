# Data manipulation and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Statistical tests
from scipy.stats import chi2_contingency, ttest_ind
import statsmodels.api as sm



class Hypotesis():
    def __init__(self, df):
        self.df=df

    def KPI(self):
        # Creating KPI columns
        self.df['Risk'] = self.df['TotalClaims'] / self.df['SumInsured']
        self.df['Profit_Margin'] = ((self.df['TotalPremium'] - self.df['TotalClaims']) / self.df['TotalPremium']) * 100

        # Summary statistics for the KPIs
        print (f"__----Summary statistics for KPIs__----- , \n {self.df[['Risk', 'Profit_Margin']].describe()}")


    def risk_by_province(self):
       # Contingency table of Risk by Province
        province_contingency_table = pd.crosstab(self.df['Province'], self.df['Risk'])

        # Performing chi-squared test
        chi2, p, dof, expected = chi2_contingency(province_contingency_table)

        # Print results
        print(f"Chi-Squared Test Results for Risk across Provinces:\nChi-Squared: {chi2}\nP-value: {p}")

        # Decision based on p-value
        if p < 0.05:
            print("Reject the null hypothesis: There is a significant difference in risk between provinces.")
        else:
            print("Fail to reject the null hypothesis: No significant difference in risk across provinces.")


    
    def risk_by_zipcode(self):

         # Contingency table of Risk by Zip Code
        zipcode_contingency_table = pd.crosstab(self.df['PostalCode'], self.df['Risk'])

        # Performing chi-squared test
        chi2_zip, p_zip, dof_zip, expected_zip = chi2_contingency(zipcode_contingency_table)

        # Print results
        print(f"Chi-Squared Test Results for Risk across Zip Codes:\nChi-Squared: {chi2_zip}\nP-value: {p_zip}")

        # Decision based on p-value
        if p_zip < 0.05:
            print("Reject the null hypothesis: There is a significant difference in risk between zip codes.")
        else:
            print("Fail to reject the null hypothesis: No significant difference in risk across zip codes.")
    

    def profit_mergin_between_zipcode(self):
        # Defining two groups for t-test
        zip_a = self.df[self.df['PostalCode'] == 'A']['Profit_Margin']
        zip_b = self.df[self.df['PostalCode'] == 'B']['Profit_Margin']

        # Performing t-test
        t_stat, p_val = ttest_ind(zip_a, zip_b)

        # Print results
        print(f"T-Test Results for Profit Margin between Zip Codes:\nT-statistic: {t_stat}\nP-value: {p_val}")

        # Decision based on p-value
        if p_val < 0.05:
            print("Reject the null hypothesis: There is a significant difference in profit margins between zip codes.")
        else:
            print("Fail to reject the null hypothesis: No significant difference in profit margins across zip codes.")
    

    def t_test_basedOn_gender(self):
        # Defining two groups for t-test based on gender
        risk_men = self.df[self.df['Gender'] == 'Male']['Risk']
        risk_women = self.df[self.df['Gender'] == 'Female']['Risk']

        # Performing t-test
        t_stat_gender, p_val_gender = ttest_ind(risk_men, risk_women)

        # Print results
        print(f"T-Test Results for Risk between Genders:\nT-statistic: {t_stat_gender}\nP-value: {p_val_gender}")

        # Decision based on p-value
        if p_val_gender < 0.05:
            print("Reject the null hypothesis: There is a significant difference in risk between men and women.")
        else:
            print("Fail to reject the null hypothesis: No significant difference in risk between men and women.")

    

    
    def key_insight_visualization(self):
        # Bar plot for average risk across provinces
        sns.barplot(x='Province', y='Risk', data=self.df)
        plt.title('Average Risk Across Provinces')
        plt.xticks(rotation=45)
        plt.show()

        # Plotting profit margin distribution across zip codes
        plt.figure(figsize=(10,6))
        sns.scatterplot(x='PostalCode', y='Profit_Margin', data=self.df, alpha=0.5)
        plt.title('Profit Margin Distribution Across Zip Codes')
        plt.xticks(rotation=90)
        plt.show()

        # Histogram for risk differences between men and women
        sns.histplot(self.df[self.df['Gender'] == 'Male']['Risk'], color='blue', label='Men', kde=True)
        sns.histplot(self.df[self.df['Gender'] == 'Female']['Risk'], color='pink', label='Women', kde=True)
        plt.legend()
        plt.title('Risk Distribution by Gender')
        plt.show()

        


    def risk_difference_accross_gender(self):

        # Plotting risk distribution by gender
        plt.figure(figsize=(8,6))
        sns.histplot(data=self.df, x='Risk', hue='Gender', element='step', bins=20, stat="count", common_norm=False)
        plt.title('Risk Distribution by Gender')
        plt.show()
