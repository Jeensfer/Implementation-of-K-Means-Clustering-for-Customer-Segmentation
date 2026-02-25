# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the customer dataset from the CSV file and select relevant features (Annual Income and Spending Score).
2. Normalize the selected features using StandardScaler.
3. Apply the K-Means algorithm and determine the optimal number of clusters using the Elbow Method.
4. Assign cluster labels to customers and visualize the segmented groups.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Jeensfer Jo
RegisterNumber:  212225240058

# Fix for MKL warning (Windows)
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print(df.head())

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Elbow Method to find optimal K
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# -----------------------------
# Apply KMeans (Assume K=5)
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# -----------------------------
# Visualize Clusters
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['Annual Income (k$)'],
            df['Spending Score (1-100)'],
            c=df['Cluster'],
            cmap='viridis')

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()
*/
```

## Output:
<img width="529" height="107" alt="Screenshot 2026-02-25 113043" src="https://github.com/user-attachments/assets/cd0c976a-a27c-43d3-9e98-18c7f3b3a689" />
<img width="639" height="476" alt="Screenshot 2026-02-25 113021" src="https://github.com/user-attachments/assets/5485c4e5-45c1-41e3-90c3-d5bda3032c2e" />
<img width="793" height="593" alt="Screenshot 2026-02-25 113030" src="https://github.com/user-attachments/assets/cf6d07b7-c1dd-49f2-b3e1-2793bdcd47dc" />





## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
