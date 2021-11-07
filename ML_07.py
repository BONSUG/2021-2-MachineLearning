#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
mnist = fetch_openml('mnist_784')

feature,target=mnist['data'],mnist['target']

x_train,x_test,y_train,y_test=train_test_split(feature,target,test_size=0.1428)
#feature[:60000],feature[60000:],target[:60000],target[60000:]
from sklearn.svm import LinearSVC 
model = LinearSVC( ) 
clf = model.fit(x_train, y_train) 
clf.predict(x_test)

import numpy as np
pred=clf.predict(x_test)
print("테스트 셑의 정확도: {:.2f}".format(np.mean(pred==y_test)))


# In[13]:


from sklearn.svm import SVC 
model = SVC(kernel='rbf', C=1, gamma=0.1) 
clf = model.fit(x_train, y_train) 
pred=clf.predict(x_test)


# In[15]:


print("테스트 셑의 정확도: {:.2f}".format(np.mean(pred==y_test)))

