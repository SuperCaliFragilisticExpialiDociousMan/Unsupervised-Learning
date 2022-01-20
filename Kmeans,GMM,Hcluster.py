#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:07:40 2021

@author: shizhengyan
"""
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

def pooling(new_list):
    # pooling
    pool=[]
    i,j=0,0
    while i < len(new_list):
        l=[]
        j=0
        while j<len(new_list[0]):
            l.append(new_list[i,j])
            j+=5
        l=np.array(l)
        pool.append(l)
        i+=5
    pool=np.array(pool)
    return pool


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range



'K-means'
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


img_1 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/3096_colorPlane.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img_1)
plt.show()

img_1=pooling(img_1)

# converts the MxNx3 image into a Kx3 matrix where K=MxN and each row is now a vector in the 3-D space of RGB.
vectorized = img_1.reshape((-1,3))

# convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
vectorized = np.float32(vectorized)

# Define criteria, number of clusters(K) and apply k-means()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


K = 5
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
print('center: ',center)
print(center.shape)
# Now convert back into uint8.
center = np.uint8(center)
print('center: ',center)
print('label: ',label)
#Next, we have to access the labels to regenerate the clustered image
res = center[label.flatten()]
result_image = res.reshape((img_1.shape))

figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
img_1 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/3096_colorPlane.jpg'), cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1),plt.imshow(img_1)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image' ), plt.xticks([]), plt.yticks([])
plt.show() 

for K in range(2,6):
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    score = silhouette_score(vectorized, np.array(label), metric='euclidean')
    score_AGclustering_c = calinski_harabasz_score(vectorized, np.array(label))
    print('K=',K,' silhouette_score=',score)
    print('K=',K,' CH index=',score_AGclustering_c)

img_2 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/42049_colorBird.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img_2)
plt.show()

img_2=pooling(img_2)

# converts the MxNx3 image into a Kx3 matrix where K=MxN and each row is now a vector in the 3-D space of RGB.
vectorized2 = img_2.reshape((-1,3))

# convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
vectorized2 = np.float32(vectorized2)

# Define criteria, number of clusters(K) and apply k-means()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


K = 5
attempts=10
ret,label,center=cv2.kmeans(vectorized2,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
print('center: ',center)
print(center.shape)
# Now convert back into uint8.
center = np.uint8(center)
print('center: ',center)
print('label: ',label)
#Next, we have to access the labels to regenerate the clustered image
res = center[label.flatten()]
result_image2 = res.reshape((img_2.shape))

figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
img_2 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/42049_colorBird.jpg'), cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1),plt.imshow(img_2)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image2)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show() 

for K in range(2,6):
    attempts=10
    ret,label,center=cv2.kmeans(vectorized2,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    score = silhouette_score(vectorized2, np.array(label), metric='euclidean')
    score_AGclustering_c = calinski_harabasz_score(vectorized2, np.array(label))
    print('K=',K,' silhouette_score=',score)
    print('K=',K,' CH index=',score_AGclustering_c)

###############################################################


'GMM'


import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

img = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/3096_colorPlane.jpg'), cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1,3))
img=vectorized
hist, bin_edges = np.histogram(img, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

classif = GaussianMixture(n_components=2)
classif.fit(img.reshape((img.size, 1)))

threshold = np.mean(classif.means_)
binary_img = img > threshold

plt.figure(figsize=(11,4))
plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.subplot(132)
plt.plot(bin_centers, hist, lw=2)
plt.axvline(0.5, color='r', ls='--', lw=2)
plt.text(0.57, 0.8, 'histogram', fontsize=20, transform = plt.gca().transAxes)
plt.yticks([])
plt.subplot(133)
plt.imshow(binary_img, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
plt.show()



import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

img = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/3096_colorPlane.jpg'), cv2.COLOR_BGR2RGB)

newdata = img.reshape(-1, 3)
gmm = GaussianMixture(n_components=2, covariance_type="tied")
gmm = gmm.fit(newdata)

cluster = gmm.predict(newdata)
cluster = cluster.reshape(321, 481)
plt.imshow(cluster)
plt.show()

img2 =  cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/42049_colorBird.jpg'), cv2.COLOR_BGR2RGB)

newdata2 = img2.reshape(-1, 3)
gmm2 = GaussianMixture(n_components=2, covariance_type="tied")
gmm2 = gmm2.fit(newdata2)

cluster2 = gmm.predict(newdata2)
cluster2 = cluster2.reshape(321, 481)
plt.imshow(cluster2)
plt.show()

##################################################################
'H-cluster'

import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from numpy import unique
image= cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/3096_colorPlane.jpg'), cv2.COLOR_BGR2RGB)

# subsampling
new_list=pooling(image)
vectorized = new_list.reshape((-1,3))

model=AgglomerativeClustering(n_clusters=2)
yhat=model.fit(vectorized)
yhat_2=model.fit_predict(vectorized)
clusters=unique(yhat)


result_image = yhat_2.reshape(len(new_list),len(new_list[0]))
plt.title('h-cluster when k=2: img1')
plt.imshow(result_image)

for i in range(2,6):
    model=AgglomerativeClustering(n_clusters=i)
    yhat=model.fit(vectorized)
    yhat_2=model.fit_predict(vectorized)
    clusters=unique(yhat)
    score_AGclustering_s = silhouette_score(vectorized, yhat.labels_, metric='euclidean')
    score_AGclustering_c = calinski_harabasz_score(vectorized, yhat.labels_)
    print('Silhouette Score: %.4f' % score_AGclustering_s)
    print('Calinski Harabasz Score: %.4f' % score_AGclustering_c)


image2= cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/42049_colorBird.jpg'), cv2.COLOR_BGR2RGB)

# pooling
new_list2=pooling(image2)
vectorized2 = new_list2.reshape((-1,3))

model=AgglomerativeClustering(n_clusters=2)
yhat=model.fit(vectorized2)
yhat_2=model.fit_predict(vectorized2)
clusters=unique(yhat)


result_image2 = yhat_2.reshape(len(new_list2),len(new_list2[0]))
plt.title('h-cluster when k=2: img2')
plt.imshow(result_image2)


figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(result_image)
plt.title('h-cluster when k=2: img1'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image2)
#plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.title('h-cluster when k=2: img2')
plt.show() 
