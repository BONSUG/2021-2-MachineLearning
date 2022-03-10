#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn.datasets import load_files

reviews_train=load_files("aclImdb/train/")
text_train,y_train=reviews_train.data,reviews_train.target
text_train = [doc.replace(b"<br />",b" ")for doc in text_train]
reviews_test=load_files("aclImdb/test/")
text_test,y_test=reviews_test.data,reviews_test.target
text_test = [doc.replace(b"<br />",b" ")for doc in text_test]


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(min_df=default).fit(text_train+text_test)
X_train=vect.transform(text_train)
X_test=vect.transform(text_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_train, y_train)))
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_test, y_test)))


# In[38]:


vect=CountVectorizer().fit(text_train+text_test)
X_train=vect.transform(text_train)
X_test=vect.transform(text_test)
logreg = LogisticRegression().fit(X_train, y_train)
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_train, y_train)))
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_test, y_test)))


# In[39]:


vect=CountVectorizer(min_df=5).fit(text_train+text_test)
X_train=vect.transform(text_train)
X_test=vect.transform(text_test)
logreg = LogisticRegression().fit(X_train, y_train)
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_train, y_train)))
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_test, y_test)))


# In[41]:


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
vect=CountVectorizer(stop_words="english").fit(text_train+text_test)
X_train=vect.transform(text_train)
X_test=vect.transform(text_test)
logreg = LogisticRegression().fit(X_train, y_train)
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_train, y_train)))
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_test, y_test)))

