from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
# Step 1: Select relevant features for segmentation
segmentation_features = ['age', 'balance', 'products_number', 'estimated_salary', 'active_member', 'churn']
# One-hot encode categorical features (e.g., gender, country)
data_encoded = pd.get_dummies(data, columns=['gender', 'country'], drop_first=True)
# Include all one-hot encoded columns dynamically in segmentation features
encoded_columns = [col for col in data_encoded.columns if 'gender_' in col or 'country_' in col]
segmentation_features.extend(encoded_columns)
# Step 2: Normalize numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_encoded[segmentation_features])
# Step 3: Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
# Step 4: Apply K-Means clustering with the optimal number of clusters
optimal_clusters = 4  # Choose based on the elbow curve
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data_encoded['Segment'] = kmeans.fit_predict(scaled_features)
# Step 5: Visualize the customer segments using PCA (for 2D projection)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
data_encoded['PCA1'] = pca_components[:, 0]
data_encoded['PCA2'] = pca_components[:, 1]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_encoded, x='PCA1', y='PCA2', hue='Segment', palette="viridis")
plt.title("Customer Segments Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Segment")
plt.show()
# Step 6: Analyze the customer segments
segment_analysis = data_encoded.groupby('Segment').mean()
print("Customer Segment Analysis:")
print(segment_analysis)
# Save the segmented dataset (optional)
data_encoded.to_csv('Customer_Segments.csv', index=False)
print("\nSegmented dataset saved as 'Customer_Segments.csv'!")
