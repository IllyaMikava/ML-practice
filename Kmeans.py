import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load your data
df = pd.read_excel('CSV/xlsx/Wk 10 student grades.xlsx')

# Select the grade columns for clustering
grade_columns = ['HW_Avg', 'Quiz_Avg', 'Exam1', 'Exam2', 'Project']
X = df[grade_columns]

# Standardize the data (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal K using elbow method
wcss = []  # Within-Cluster Sum of Square
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

# Apply K-means with chosen K (let's use K=3 based on elbow)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = clusters

# Use PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Create the cluster visualization
plt.figure(figsize=(12, 8))

# Scatter plot with clusters
scatter = sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', 
                         palette='viridis', s=100, alpha=0.8)

# Add cluster centers in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', 
           marker='X', s=200, label='Cluster Centers', edgecolors='black')

plt.title(f'Student Grade Clusters (K={k})', fontsize=16, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print cluster statistics
print("=" * 50)
print(f"CLUSTER ANALYSIS (K={k})")
print("=" * 50)

for cluster_num in range(k):
    cluster_data = df[df['Cluster'] == cluster_num]
    print(f"\nCluster {cluster_num} - {len(cluster_data)} students")
    print("Average Grades:")
    for col in grade_columns:
        avg_grade = cluster_data[col].mean()
        print(f"  {col}: {avg_grade:.1f}")

# Additional visualization: Pairplot to see relationships between original features
plt.figure(figsize=(12, 8))
sns.pairplot(df, vars=grade_columns, hue='Cluster', palette='viridis', 
             diag_kind='hist', corner=True)
plt.suptitle('Pairwise Relationships Colored by Cluster', y=1.02)
plt.show()

# Bar plot showing average grades by cluster
cluster_means = df.groupby('Cluster')[grade_columns].mean()

plt.figure(figsize=(12, 6))
cluster_means.plot(kind='bar', figsize=(12, 6))
plt.title('Average Grades by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Grade')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Display some students from each cluster
print("\n" + "=" * 50)
print("SAMPLE STUDENTS FROM EACH CLUSTER")
print("=" * 50)

for cluster_num in range(k):
    print(f"\nCluster {cluster_num} Students (sample):")
    cluster_students = df[df['Cluster'] == cluster_num].head(3)
    for _, student in cluster_students.iterrows():
        print(f"  Student {student['Student']}: HW={student['HW_Avg']}, "
              f"Quiz={student['Quiz_Avg']}, Exam1={student['Exam1']}")
        
# Visualize clusters using two actual grade dimensions
plt.figure(figsize=(12, 8))

# Using HW_Avg and Exam1 for visualization
scatter = sns.scatterplot(data=df, x='HW_Avg', y='Exam1', hue='Cluster', 
                         palette='viridis', s=100, alpha=0.8)

plt.title(f'Student Clusters by Homework vs Exam 1 Performance', fontsize=16)
plt.xlabel('Homework Average')
plt.ylabel('Exam 1 Score')
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)

# Annotate some points with student IDs
for i in range(min(10, len(df))):
    plt.annotate(str(df.iloc[i]['Student']), 
                (df.iloc[i]['HW_Avg'], df.iloc[i]['Exam1']),
                textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

plt.show()

# Create a summary table of cluster characteristics
cluster_summary = df.groupby('Cluster').agg({
    'HW_Avg': ['mean', 'std'],
    'Quiz_Avg': ['mean', 'std'], 
    'Exam1': ['mean', 'std'],
    'Exam2': ['mean', 'std'],
    'Project': ['mean', 'std'],
    'Student': 'count'
}).round(1)

print("\n" + "=" * 60)
print("CLUSTER SUMMARY STATISTICS")
print("=" * 60)
print(cluster_summary)