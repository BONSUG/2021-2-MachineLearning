#!/usr/bin/env python
# coding: utf-8


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


data = np.loadtxt("C:\\Users\\qhstj\\Downloads\\iris.csv", delimiter=",", dtype=np.float32)
train_input,test_input,train_target,test_target=train_test_split(data[:,:-1],data[:,-1:],random_state=0)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(train_input,train_target)
print("K=1일 때 테스트 셀의 정확도 : "+ str(kn.score(test_input,test_target)))

kn1 = KNeighborsClassifier(n_neighbors=5)
kn1.fit(train_input,train_target)
print("K=5일 때 테스트 셀의 정확도 : "+ str(kn1.score(test_input,test_target)))

kn2 = KNeighborsClassifier(n_neighbors=10)
kn2.fit(train_input,train_target)
print("K=10일 때 테스트 셀의 정확도 : "+ str(kn2.score(test_input,test_target)))



data1 = np.loadtxt("C:\\Users\\qhstj\\Downloads\\iris_mod.csv", delimiter=",", dtype=np.float32)

train_input,test_input,train_target,test_target=train_test_split(data1[:,:-1],data1[:,-1:],random_state=0)
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(train_input,train_target)
y_pred = kn.predict(test_input)
print("k=1일 때 normaliation 수행 전 테스트 셑의 정확도: {:.2f}".format(np.mean(y_pred==test_target)))

norm = (data1 - data1.mean(axis=0))/data1.std(axis=0)
norm= norm.astype(np.int)

train_input1,test_input1,train_target1,test_target1=train_test_split(norm[:,:-1],norm[:,-1:],random_state=0)
kn1 = KNeighborsClassifier(n_neighbors=1)
kn1.fit(train_input1,train_target1)
y_pred1 = kn1.predict(test_input1)
print("k=1일 때 normaliation 수행 후 테스트 셑의 정확도: {:.2f}".format(np.mean(y_pred1==test_target1)))

