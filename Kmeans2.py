import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load your data
df = pd.read_excel('Wk 10 student grades.xlsx')

# Calculate overall average grade (excluding HW to compare against it)
df['Overall_Avg'] = (df['Quiz_Avg'] + df['Exam1'] + df['Exam2'] + df['Project']) / 4

# Prepare data for clustering (using HW_Avg vs Overall_Avg)
X = df[['HW_Avg', 'Overall_Avg']]

# Apply K-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Create the simple 2D plot
plt.figure(figsize=(12, 8))

# Scatter plot colored by clusters
scatter = sns.scatterplot(data=df, x='HW_Avg', y='Overall_Avg', hue='Cluster', 
                         palette='viridis', s=100, alpha=0.8)

# Add cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, 
           label='Cluster Centers', edgecolors='black')

# Add student labels
for i in range(len(df)):
    plt.annotate(str(df.iloc[i]['Student']), 
                (df.iloc[i]['HW_Avg'], df.iloc[i]['Overall_Avg']),
                textcoords="offset points", xytext=(5,5), ha='left', 
                fontsize=8, alpha=0.7)

plt.title('Student Clusters: Homework vs Overall Performance', fontsize=16, fontweight='bold')
plt.xlabel('Homework Average')
plt.ylabel('Overall Average (Quizzes + Exams + Project)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add reference line (y = x) to show students performing equally in both
max_grade = max(df[['HW_Avg', 'Overall_Avg']].max())
plt.plot([0, max_grade], [0, max_grade], 'r--', alpha=0.3, label='Equal Performance Line')

plt.legend()
plt.tight_layout()
plt.show()

# Print simple cluster descriptions
print("=" * 60)
print("CLUSTER DESCRIPTIONS")
print("=" * 60)

for cluster_num in range(3):
    cluster_data = df[df['Cluster'] == cluster_num]
    hw_avg = cluster_data['HW_Avg'].mean()
    overall_avg = cluster_data['Overall_Avg'].mean()
    performance_gap = hw_avg - overall_avg
    
    print(f"\nCluster {cluster_num} - {len(cluster_data)} students")
    print(f"Average Homework: {hw_avg:.1f}")
    print(f"Average Overall: {overall_avg:.1f}")
    
    if performance_gap > 2:
        print("Profile: Stronger in Homework")
    elif performance_gap < -2:
        print("Profile: Stronger in Exams/Quizzes") 
    else:
        print("Profile: Balanced Performance")