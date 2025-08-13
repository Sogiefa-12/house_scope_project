import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class DataAnalysisApp:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
    
    def clean_data(self):
        print(f"Original shape: {self.df.shape}")
        
        # Fill missing numeric values with median
        num_cols = self.df.select_dtypes(include=np.number).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())

        # Fill missing categorical values with mode
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # Remove numeric outliers using IQR
        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        print(f"After cleaning shape: {self.df.shape}")
    
    def descriptive_stats(self):
        print("\n--- Numerical Data Stats ---")
        print(self.df.describe())
        print("\n--- Categorical Data Stats ---")
        print(self.df.describe(include=['object']))
    
    def visualize_data(self):
        # Bar chart for neighborhood counts
        plt.figure(figsize=(10, 5))
        sns.countplot(x='Neighborhood', data=self.df, order=self.df['Neighborhood'].value_counts().index)
        plt.xticks(rotation=90)
        plt.title('Number of Houses per Neighborhood')
        plt.show()

        # Histogram of SalePrice
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df['SalePrice'], bins=30, kde=True)
        plt.title('Distribution of Sale Prices')
        plt.show()

        # Scatter plot between GrLivArea and SalePrice
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='GrLivArea', y='SalePrice', data=self.df)
        plt.title('Living Area vs. Sale Price')
        plt.show()
    
    def hypothesis_tests(self):
        # T-test example: compare SalePrice between two neighborhoods
        neighborhoods = self.df['Neighborhood'].unique()
        if len(neighborhoods) >= 2:
            group1 = self.df[self.df['Neighborhood'] == neighborhoods[0]]['SalePrice']
            group2 = self.df[self.df['Neighborhood'] == neighborhoods[1]]['SalePrice']
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
            print(f"\nT-test between {neighborhoods[0]} and {neighborhoods[1]} SalePrice:")
            print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
        
        # Chi-square test: Neighborhood vs HouseStyle
        contingency_table = pd.crosstab(self.df['Neighborhood'], self.df['HouseStyle'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-square test between Neighborhood and HouseStyle:")
        print(f"chi2 = {chi2:.3f}, p-value = {p:.3f}, degrees of freedom = {dof}")

if __name__ == "__main__":
    app = DataAnalysisApp('train.csv')  # Kaggle Ames Housing train.csv
    app.clean_data()
    app.descriptive_stats()
    app.visualize_data()
    app.hypothesis_tests()
