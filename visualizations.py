import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder for saving visualizations
output_dir = './visualizations_output'
os.makedirs(output_dir, exist_ok=True)

# Load the cleaned data
data = pd.read_csv('./Dataset/patient_data_cleaned.csv')

# Set the style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================
# 1. Gender Distribution - Bar Chart
# ============================================
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Gender", palette="Set2")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f'{output_dir}/01_gender_distribution_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 2. Gender Distribution - Pie Chart
# ============================================
plt.figure(figsize=(8, 4))
data['Gender'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(5, 5), colors=["#6DD5FF", "#90F79A"])
plt.title("Gender Distribution (Pie Chart)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f'{output_dir}/02_gender_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 3. Hypertension Stages Distribution
# ============================================
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Stages", palette="coolwarm")
plt.title("Hypertension Stages Distribution")
plt.xlabel("Stages")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_hypertension_stages.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 4. Correlation between Systolic and Diastolic
# ============================================
# Heatmap on encoded numeric BP
plt.figure(figsize=(6, 4))
correlation_cols = ['Systolic', 'Diastolic']
sns.heatmap(data[correlation_cols].corr(), annot=True, cmap="Blues")
plt.title("Correlation between Systolic & Diastolic")
plt.tight_layout()
plt.savefig(f'{output_dir}/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 5. TakeMedication vs Severity
# ============================================
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="TakeMedication", hue="Severity", palette="Set1")
plt.title("TakeMedication vs Severity")
plt.xlabel("TakeMedication")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f'{output_dir}/05_medication_vs_severity.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 6. Age Group vs Hypertension Stages
# ============================================
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x="Age", hue="Stages", palette="husl")
plt.title("Age Group vs Hypertension Stages")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f'{output_dir}/06_age_vs_stages.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 7. Pairplot: Systolic vs Diastolic across Stages
# ============================================
pairplot = sns.pairplot(data[["Systolic", "Diastolic", "Stages"]], hue="Stages", diag_kind="kde", palette="husl")
pairplot.fig.suptitle("Pairplot: Systolic vs Diastolic across Stages", y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/07_pairplot_systolic_diastolic.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 8. Distribution of Severity
# ============================================
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Severity", palette="RdYlGn_r")
plt.title("Distribution of Severity")
plt.xlabel("Severity Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f'{output_dir}/08_severity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 9. Blood Pressure Systolic Distribution
# ============================================
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Systolic", palette="viridis")
plt.title("Blood Pressure - Systolic Distribution")
plt.xlabel("Systolic Range")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/09_systolic_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 10. Blood Pressure Diastolic Distribution
# ============================================
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Diastolic", palette="plasma")
plt.title("Blood Pressure - Diastolic Distribution")
plt.xlabel("Diastolic Range")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/10_diastolic_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ All visualizations generated successfully!")
print(f"Visualizations saved to: {output_dir}/")

