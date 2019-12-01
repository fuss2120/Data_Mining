#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;


# In[2]:


data = pd.read_csv('interview3.csv', error_bad_lines=False);


# In[3]:


data_text = data[['Summary']];


# In[4]:


data_text = data_text.astype('str');


# In[5]:


from nltk.corpus import stopwords;
import nltk;


# In[6]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


# In[7]:


for idx in range(len(data_text)):
    
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    data_text.iloc[idx]['Summary'] = [word for word in data_text.iloc[idx]['Summary'].split(' ') 
                                         if word not in stopwords.words()];
    
    #print logs to monitor output
    if idx % 1000 == 0:
        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data_text)));


# In[8]:


train_transcript = [value[0] for value in data_text.iloc[0:].values];


# In[9]:


num_topics = 10;


# In[10]:


#the count vectorizer needs string inputs, not array, so I join them with a space.
train_transcript_sentences = [' '.join(text) for text in train_transcript]


# In[11]:


vectorizer = CountVectorizer(analyzer='word', max_features=5000);
x_counts = vectorizer.fit_transform(train_transcript_sentences);


# In[12]:


transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);


# In[13]:


xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)


# In[14]:


model = NMF(n_components=num_topics, init='nndsvd');


# In[15]:


model.fit(xtfidf_norm)


# In[16]:


def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);


# In[17]:


get_nmf_topics(model, 20)


# In[ ]:





# In[ ]:





# In[ ]:




