#!/usr/bin/env python
# coding: utf-8

# In[3]:


num = input("n? ")
n=int(num)
sum=0
i=1
while i<= n:
    if i%2==0:
          sum+=i
    i+=1
    
print(sum)

