#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd
import numpy as np

head =['Sepal length','Sepal width','Petal length','Petal Width','Speices']
df=pd.read_csv('C:\\Users\\qhstj\\Downloads\\small_iris.csv',names=head)
for i in range(1,4) :
    class1=df[df['Speices']==i]
    print('클래스 '+str(i)+'의 평균벡터는 '+str(np.around(np.array(class1.mean()[0:4]),1)))

