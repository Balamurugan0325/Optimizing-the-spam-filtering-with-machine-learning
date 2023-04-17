#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[2]:


df = pd.read_csv("spam.csv",encoding="latin-1")
df.head()


# In[3]:


df.info()


# In[4]:


df.isna().sum()


# In[5]:


df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)


# In[6]:


df.rename(columns = {"v1":"label", "v2":"message"}, inplace = True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# In[7]:


df.tail()


# In[8]:


import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[9]:


corpus = []
ps = PorterStemmer()


# In[10]:


for sms_string in list(df.message):
   message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)
   message = message.lower()
   words = message.split()
   words = [word for word in words if word not in set(stopwords.words('english'))]
   words = [ps.stem(word) for word in words]
   message = ' '.join(words)
   corpus.append(message)


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()


# In[12]:


y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values


# In[13]:


pickle.dump(cv, open('cv-transform.pkl', 'wb'))


# In[14]:


df.describe()


# In[15]:


df.shape


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[17]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.3)
classifier.fit(X_train, y_train)


# In[18]:


filename = 'spam-sms-mnb-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:




