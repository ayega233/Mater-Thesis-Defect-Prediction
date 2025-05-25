import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# Step 1: Prepare the Data (Simulated Data for 23 repositories)
data = pd.read_csv('D:/MSC/SEMESTER3/TextMining/Labs/myenv/statistical/individual.csv')

df = pd.DataFrame(data)

# Step 2: Compute Correlation Coefficients
metrics = ['validation precision', 'validation recall', 'validation f1-score', 'valiadation_accuracy']
correlations = {}
p_values = {}

for metric in metrics:
    # Pearson correlation between Defect_Count and the metric
    corr_defect, p_defect = pearsonr(df['Validation Defects'], df[metric])
    # Pearson correlation between NonDefect_Count and the metric
    corr_nondefect, p_nondefect = pearsonr(df['Validation Non Defect'], df[metric])
    
    correlations[f'Defect_Count vs {metric}'] = corr_defect
    correlations[f'NonDefect_Count vs {metric}'] = corr_nondefect
    p_values[f'Defect_Count vs {metric}'] = p_defect
    p_values[f'NonDefect_Count vs {metric}'] = p_nondefect

# Step 3: Print correlation coefficients and p-values
print("Correlation Coefficients and p-values:")
for key in correlations.keys():
    print(f"{key}: Correlation = {correlations[key]:.4f}, p-value = {p_values[key]:.4f}")

# Step 4: Multiple Testing Correction (Benjamini-Hochberg)
p_values_list = list(p_values.values())
reject, pvals_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='fdr_bh')

# Print corrected p-values
print("\nCorrected p-values (Benjamini-Hochberg):")
for i, key in enumerate(p_values.keys()):
    print(f"{key}: Corrected p-value = {pvals_corrected[i]:.4f}, Reject Null Hypothesis = {reject[i]}")
