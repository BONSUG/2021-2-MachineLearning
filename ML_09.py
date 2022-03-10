#!/usr/bin/env python
# coding: utf-8

# In[174]:


from sklearn.datasets import load_files
import chardet

files=load_files("C:\\Users\\qhstj\\Downloads\\bbcsport\\")


# In[175]:


X, y = files.data, files.target
for i in range(len(X)): 
    if(chardet.detect(X[i])!="utf-8"): 
        X[i]=X[i].decode(chardet.detect(X[i])['encoding']).encode('utf8') 
# 데이터 전처리
X = [doc.replace(b"",b"") for doc in X] 
X = [doc.replace(b"\n",b" ") for doc in X]


# In[176]:


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',norm='l2')
X = vectorizer.fit_transform(X)


# In[177]:


print(X.shape)


# In[178]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target=train_test_split(X,y,random_state=0)
import numpy as np
scores=cross_val_score(KNeighborsClassifier(n_neighbors=1),train_input,train_target,cv=5)
print("크로스 밸리데이션 평균 점수 : {:.2f}".format(np.mean(scores)))


# In[179]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X=X.toarray()
pca.fit(X)
X_pca=pca.transform(X)
train_input,test_input,train_target,test_target=train_test_split(X_pca,y,random_state=0)
scores=cross_val_score(KNeighborsClassifier(n_neighbors=1),train_input,train_target,cv=5)
print("크로스 밸리데이션 평균 점수 : {:.2f}".format(np.mean(scores)))


# In[180]:


pca=PCA(n_components=10)
pca.fit(X)
X_pca=pca.transform(X)
train_input,test_input,train_target,test_target=train_test_split(X_pca,y,random_state=0)
scores=cross_val_score(KNeighborsClassifier(n_neighbors=1),train_input,train_target,cv=5)
print("크로스 밸리데이션 평균 점수 : {:.2f}".format(np.mean(scores)))

