#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:07:40 2021

@author: shizhengyan
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

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

img_1 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/3096_colorPlane.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img_1)
plt.show()


new_list=[]
for i in range(len(img_1)):
    for j in range(len(img_1[0])):
        l=[]
        l.append(i)
        l.append(j)
        l.append(img_1[i][j][0])
        l.append(img_1[i][j][1])
        l.append(img_1[i][j][2])
        new_list.append(l)
new_list=np.array(new_list)
img_1=new_list

img_2 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/42049_colorBird.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img_2)
plt.show()

new_list=[]
for i in range(len(img_2)):
    for j in range(len(img_2[0])):
        l=[]
        l.append(i)
        l.append(j)
        l.append(img_2[i][j][0])
        l.append(img_2[i][j][1])
        l.append(img_2[i][j][2])
        new_list.append(l)

new_list=np.array(new_list)

img_2=new_list

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

img_1[:,0]=normalization(img_1[:,0])
img_1[:,1]=normalization(img_1[:,1])
img_1[:,2]=normalization(img_1[:,2])
img_1[:,3]=normalization(img_1[:,3])
img_1[:,4]=normalization(img_1[:,4])

img_2[:,0]=normalization(img_2[:,0])
img_2[:,1]=normalization(img_2[:,1])
img_2[:,2]=normalization(img_2[:,2])
img_2[:,3]=normalization(img_2[:,3])
img_2[:,4]=normalization(img_2[:,4])

from sklearn.decomposition import PCA

pca_img1 = PCA(n_components=5)
pca_img1.fit(img_1.T)
trans_pca_img1 = pca_img1.transform(img_1.T)
pca_img2 = PCA(n_components=5)
pca_img2.fit(img_2.T)
trans_pca_img2 = pca_img1.transform(img_2.T)

print(f"img 1 : {sum(pca_img1.explained_variance_ratio_)}")
print(f"img 2: {sum(pca_img2.explained_variance_ratio_)}")

list1=pca_img1.explained_variance_ratio_.tolist()
list2=pca_img2.explained_variance_ratio_.tolist()


from matplotlib.ticker import PercentFormatter

df = pd.DataFrame({'Eigen Value': list1})
df.index = ['eigenvalue1', 'eigenvalue2', 'eigenvalue3', 'eigenvalue4', 'eigenvalue5']
df = df.sort_values(by='Eigen Value',ascending=False)
df["cumpercentage"] = df["Eigen Value"].cumsum()/df["Eigen Value"].sum()*100
fig, ax = plt.subplots()
ax.bar(df.index, df["Eigen Value"], color="C0")
ax2 = ax.twinx()
ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.show()



df = pd.DataFrame({'Eigen Value': list2})
df.index = ['eigenvalue1', 'eigenvalue2', 'eigenvalue3', 'eigenvalue4', 'eigenvalue5']
df = df.sort_values(by='Eigen Value',ascending=False)
df["cumpercentage"] = df["Eigen Value"].cumsum()/df["Eigen Value"].sum()*100
fig, ax = plt.subplots()
ax.bar(df.index, df["Eigen Value"], color="C0")
ax2 = ax.twinx()
ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.show()


'Kmeans return to PCA'

pca_img1 = PCA(n_components=2)
pca_img1.fit(img_1.T)
trans_pca_img1 = pca_img1.transform(img_1.T)
components=pca_img1.components_
components=components.T
print(components)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(components)
label = kmeans.fit_predict(components)

#filter rows of original data
filtered_label0 = components[label == 0]
filtered_label1 = components[label == 1]
filtered_label2 = components[label == 2]
filtered_label3 = components[label == 3]
filtered_label4 = components[label == 4]

#plotting the results
plt.title('kmeans return to PCA: img1')
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'blue')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'red')
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'black')
plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'green')
plt.scatter(filtered_label4[:,0] , filtered_label4[:,1] , color = 'orange')
plt.show()


pca_img2 = PCA(n_components=2)
pca_img2.fit(img_2.T)
trans_pca_img2 = pca_img1.transform(img_2.T)
components=pca_img2.components_
components=components.T
print(components)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(components)
label = kmeans.fit_predict(components)

#filter rows of original data
filtered_label0 = components[label == 0]
filtered_label1 = components[label == 1]
filtered_label2 = components[label == 2]
filtered_label3 = components[label == 3]
filtered_label4 = components[label == 4]

#plotting the results
plt.title('kmeans return to PCA: img2')
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'blue')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'red')
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'black')
plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'green')
plt.scatter(filtered_label4[:,0] , filtered_label4[:,1] , color = 'orange')
plt.show()



'GMM return to PCA'
from sklearn.mixture import GaussianMixture

pca_img1 = PCA(n_components=2)
pca_img1.fit(img_1.T)
trans_pca_img1 = pca_img1.transform(img_1.T)
components=pca_img1.components_
components=components.T
print('components: ',components)
print('components[:, 0].shape: ',components[:, 0].shape)

classif = GaussianMixture(n_components=4)
classif.fit(components)
labels = classif.predict(components)
plt.title('GMM return to PCA: img1')
plt.scatter(components[:, 0], components[:, 1], c=labels, s=40, cmap='viridis');
plt.show()

pca_img2 = PCA(n_components=2)
pca_img2.fit(img_2.T)
trans_pca_img1 = pca_img1.transform(img_2.T)
components=pca_img2.components_
components=components.T
#print(components)


classif = GaussianMixture(n_components=4)
classif.fit(components)
labels = classif.predict(components)
plt.title('GMM return to PCA: img2')
plt.scatter(components[:, 0], components[:, 1], c=labels, s=40, cmap='viridis');
plt.show()


'Hierarchical cluster return to PCA'

from sklearn.cluster import AgglomerativeClustering

img_1 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/3096_colorPlane.jpg'), cv2.COLOR_BGR2RGB)
img_2 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/42049_colorBird.jpg'), cv2.COLOR_BGR2RGB)

img_1=pooling(img_1)
img_2=pooling(img_2)

new_list=[]
for i in range(len(img_1)):
    for j in range(len(img_1[0])):
        l=[]
        l.append(i)
        l.append(j)
        l.append(img_1[i][j][0])
        l.append(img_1[i][j][1])
        l.append(img_1[i][j][2])
        new_list.append(l)
new_list=np.array(new_list)
img_1=new_list

new_list=[]
for i in range(len(img_2)):
    for j in range(len(img_2[0])):
        l=[]
        l.append(i)
        l.append(j)
        l.append(img_2[i][j][0])
        l.append(img_2[i][j][1])
        l.append(img_2[i][j][2])
        new_list.append(l)
new_list=np.array(new_list)
img_2=new_list

pca_img1 = PCA(n_components=2)
pca_img1.fit(img_1.T)
trans_pca_img1 = pca_img1.transform(img_1.T)
components=pca_img1.components_
components=components.T

model=AgglomerativeClustering(n_clusters=2)
yhat=model.fit(components)
yhat_2=model.fit_predict(components)
plt.title('hierarchical return to PCA: img1')
plt.scatter(components[:, 0], components[:, 1], c=yhat_2, s=40, cmap='viridis');
plt.show()

pca_img2 = PCA(n_components=2)
pca_img2.fit(img_2.T)
trans_pca_img1 = pca_img2.transform(img_2.T)
components=pca_img2.components_
components=components.T

model=AgglomerativeClustering(n_clusters=2)
yhat=model.fit(components)
yhat_2=model.fit_predict(components)
plt.title('hierarchical return to PCA: img2')
plt.scatter(components[:, 0], components[:, 1], c=yhat_2, s=40, cmap='viridis');
plt.show()




####################################################################

img_1 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/3096_colorPlane.jpg'), cv2.COLOR_BGR2RGB)
img_2 = cv2.cvtColor(cv2.imread('/users/shizhengyan/Desktop/Neu/DS5230/week4/imageFiles/42049_colorBird.jpg'), cv2.COLOR_BGR2RGB)

from sklearn.manifold import TSNE

img_1=pooling(img_1)
img_2=pooling(img_2)

new_list_1=[]
for i in range(len(img_1)):
    for j in range(len(img_1[0])):
        l=[]
        l.append(i) # x-axis
        l.append(j) # y-axis
        l.append(img_1[i][j][0]) # red
        l.append(img_1[i][j][1]) # green
        l.append(img_1[i][j][2]) # yellow
        new_list_1.append(l)

new_list_1=np.array(new_list_1)

tsne_data_1 = TSNE(n_components=2,perplexity=30.0).fit_transform(new_list_1)
tsne_dataframe_1=pd.DataFrame({"tsne-2d-one":tsne_data_1[:,0],
                             "tsne-2d-two":tsne_data_1[:,1]})


sns.scatterplot(data=tsne_dataframe_1,
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    legend="full",
    alpha=0.3,
)
plt.title('tsne for image1 with perplexity=30')
plt.show()  # image1 of tsne

tsne_data_1 = TSNE(n_components=2,perplexity=50.0).fit_transform(new_list_1)
tsne_dataframe_1=pd.DataFrame({"tsne-2d-one":tsne_data_1[:,0],
                             "tsne-2d-two":tsne_data_1[:,1]})


sns.scatterplot(data=tsne_dataframe_1,
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    legend="full",
    alpha=0.3,
)
plt.title('tsne for image1 with perplexity=50')
plt.show()  # image1 of tsne

new_list_2=[]
for i in range(len(img_2)):
    for j in range(len(img_2[0])):
        l=[]
        l.append(i)
        l.append(j)
        l.append(img_2[i][j][0])
        l.append(img_2[i][j][1])
        l.append(img_2[i][j][2])
        new_list_2.append(l)

new_list_2=np.array(new_list_2)



tsne_data_2 = TSNE(n_components=2,perplexity=30.0).fit_transform(new_list_2)
tsne_dataframe_2=pd.DataFrame({"tsne-2d-one":tsne_data_2[:,0],
                             "tsne-2d-two":tsne_data_2[:,1]})


sns.scatterplot(data=tsne_dataframe_2,
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    legend="full",
    alpha=0.3,
)
plt.show()   # image2 of tsne


'Kmeans return to tsne'

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(tsne_data_1)
label = kmeans.fit_predict(tsne_data_1)

#filter rows of original data
filtered_label0 = tsne_data_1[label == 0]
filtered_label1 = tsne_data_1[label == 1]
filtered_label2 = tsne_data_1[label == 2]
filtered_label3 = tsne_data_1[label == 3]
filtered_label4 = tsne_data_1[label == 4]

#plotting the results
plt.title('kmeans return to tsne: img1')
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'blue')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'red')
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'black')
plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'green')
plt.scatter(filtered_label4[:,0] , filtered_label4[:,1] , color = 'orange')
plt.show()


kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(tsne_data_2)
label = kmeans.fit_predict(tsne_data_2)

#filter rows of original data
filtered_label0 = tsne_data_2[label == 0]
filtered_label1 = tsne_data_2[label == 1]
filtered_label2 = tsne_data_2[label == 2]
filtered_label3 = tsne_data_2[label == 3]
filtered_label4 = tsne_data_2[label == 4]

#plotting the results
plt.title('kmeans return to tsne: img2')
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'blue')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'red')
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'black')
plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'green')
plt.scatter(filtered_label4[:,0] , filtered_label4[:,1] , color = 'orange')
plt.show()



'GMM return to tsne'

classif = GaussianMixture(n_components=4)
classif.fit(tsne_data_1)
labels = classif.predict(tsne_data_1)
plt.title('GMM return to tsne: img1')
plt.scatter(tsne_data_1[:, 0], tsne_data_1[:, 1], c=labels, s=40, cmap='viridis');
plt.show()



classif = GaussianMixture(n_components=4)
classif.fit(tsne_data_2)
labels = classif.predict(tsne_data_2)
plt.title('GMM return to tsne: img2')
plt.scatter(tsne_data_2[:, 0], tsne_data_2[:, 1], c=labels, s=40, cmap='viridis');
plt.show()




'Hierarchical cluster return to tsne'

from sklearn.cluster import AgglomerativeClustering

model=AgglomerativeClustering(n_clusters=2)
yhat=model.fit(tsne_data_1)
yhat_2=model.fit_predict(tsne_data_1)
plt.title('hierarchical return to tsne: img1')
plt.scatter(tsne_data_1[:, 0], tsne_data_1[:, 1], c=yhat_2, s=40, cmap='viridis');
plt.show()


model=AgglomerativeClustering(n_clusters=2)
yhat=model.fit(tsne_data_2)
yhat_2=model.fit_predict(tsne_data_2)
plt.title('hierarchical return to tsne: img2')
plt.scatter(tsne_data_2[:, 0], tsne_data_2[:, 1], c=yhat_2, s=40, cmap='viridis');
plt.show()


