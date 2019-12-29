vf# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
from skimage import io
from sklearn.cluster import KMeans



image = io.imread('nass.jpg')
io.imshow(image)
io.show()
rows = image.shape[0]
cols = image.shape[1]
image = image/255
X = image.reshape(image.shape[0]*image.shape[1],3)

#from sklearn.cluster import KMeans

"""wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()"""

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 15, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
y1=kmeans.predict(X)      # k value index wise
m1=kmeans.cluster_centers_     # k cluster centeroids

X_recovered=m1[y1]
X_recovered = np.reshape(X_recovered, (rows, cols, 3))    #converting back to original 3d form
print(np.shape(X_recovered))

import imageio
imageio.imwrite('nas1.jpg', X_recovered)    #writing image


image_compressed = io.imread('nas1.jpg')        #reading image
io.imshow(image_compressed)
io.show()
