import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Data Collection and Analysis

# loading the data from csv file to a pandas dataframe
customer_data = pd.read_csv('D:/01Old/Python/Projects/personal project/archive/Mall_Customers.csv')
# print(customer_data.head())

# no. of rows and columns 
# print(customer_data.shape)

# getting more information about the dataset
# print(customer_data.info())

# checking for missing values 
customer_data.isnull().sum()

# Choosing the Annual Income Column & Speding Score Column
x = customer_data.iloc[:,[3,4]].values
# print(x)

# Choosing the correcct number of clusters

# wcSS -> within Clusters sum of squares

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_ )

# plot an elbow graph 

sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
# plt.show()
 
 # Optimum number of clusters = 5
#  Traning the K-Means Clustering Model

kmeans = KMeans(n_clusters=5,init='k-means++', random_state=0)

# return a lebel for each data point based on their cluster
Y = kmeans.fit_predict(x)
# print(Y)

# visualizing all the clusters
 
# 5 Clusters - 0,1,2,3,4
# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(x[Y==0,0], x[Y==0,1], s=50, c ='green', label='Cluster 1') 
plt.scatter(x[Y==1,0], x[Y==1,1], s=50, c ='red', label='Cluster 2') 
plt.scatter(x[Y==2,0], x[Y==2,1], s=50, c ='blue', label='Cluster 3') 
plt.scatter(x[Y==3,0], x[Y==3,1], s=50, c ='purple', label='Cluster 4') 
plt.scatter(x[Y==4,0], x[Y==4,1], s=50, c ='pink', label='Cluster 5') 

# plot the centroids 
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

