#!/usr/bin/env python
# coding: utf-8

# In[74]:


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("C:\\Users\\qhstj\\Downloads\\wdbc.data",header=None)
target=data.loc[:,1]
feature=data.loc[:,2:]
x_train,x_test,y_train,y_test=train_test_split(feature,target,random_state=0)


# In[79]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5, shuffle=True, random_state=0)
from sklearn.model_selection import cross_val_score


tree = DecisionTreeClassifier()
forest1 = RandomForestClassifier(n_estimators=1, random_state=0)
forest2 = RandomForestClassifier(n_estimators=100, random_state=0)

score1 = cross_val_score(tree, feature, target, cv=kfold)
print("교차 검증 점수1 :", score1)
print("교차 검증 점수1 평균 :", score1.mean())
score2 = cross_val_score(forest1, feature, target, cv=kfold)
print("교차 검증 점수2:", score2)
print("교차 검증 점수2 평균:", score2.mean())
score3 = cross_val_score(forest2, feature, target, cv=kfold)
print("교차 검증 점수3:", score3)
print("교차 검증 점수3 평균:", score3.mean())


# In[80]:


forest2.fit(x_train,y_train)
y_pred2=forest2.predict(x_test)
print("테스트 셑의 정확도: {:.2f}".format(np.mean(y_pred2==y_test)))

