#!/usr/bin/env python
# coding: utf-8

# In[136]:


from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
data= np.loadtxt("C:\\Users\\qhstj\\Downloads\\iris.csv", delimiter=",",skiprows=1, dtype=np.float32)


red = data[data[:,-1]==1]
green = data[data[:,-1]==2]
blue = data[data[:,-1]==3]

plt.scatter(red[:,0],red[:,1],color='r')
plt.scatter(green[:,0],green[:,1],color='g')
plt.scatter(blue[:,0],blue[:,1],color='b')

plt.show()


# In[137]:


feature,target=data[:,:-2],data[:,-1]
model = KMeans(n_clusters=3,random_state=0)
model.fit(feature)
red=feature[model.predict(feature)==0]
green=feature[model.predict(feature)==1]
blue=feature[model.predict(feature)==2]

plt.scatter(red[:,0],red[:,1],color='r')
plt.scatter(green[:,0],green[:,1],color='g')
plt.scatter(blue[:,0],blue[:,1],color='b')

plt.show()


# In[138]:


from sklearn.metrics.cluster import adjusted_rand_score
for i in range(2,7) :
    model = KMeans(n_clusters=i,random_state=0)
    model.fit(feature)
    labels = model.predict(feature)
    print("n clusters가 "+str(i)+"일 때 : "+str(adjusted_rand_score(target,labels)))

