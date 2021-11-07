#!/usr/bin/env python
# coding: utf-8

# In[28]:


student = {"국어":85,"영어":70,"수학":63,"과학":59,"사회":100}

print("국어 : "+str(student["국어"]))
print("영어 : "+str(student["영어"]))
print("수학 : "+str(student["수학"]))
print("과학 : "+str(student["과학"]))
print("사회 : "+str(student["사회"]))
print()
max=student["국어"]
n="국어"
for key in student :
    if student[key]>max :
        max=student[key]
        n=key
print("가장 점수가 높은 과목과 성적 : "+key+" "+str(max))
print()
cnt=0
total=0
for key in student :
    total=total+student[key]
    cnt=cnt+1
        
print("평균 : "+str((total/cnt)))

