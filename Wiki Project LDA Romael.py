#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;


# In[12]:


stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[13]:


num_topics=10;


# In[14]:


def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);


# In[28]:


data = pd.read_excel('CitizenActivist.xlsx', dtype={'Summary':str})
data_text = data[['Summary']];
data_text.head()


# In[29]:


processed_text = data_text['Summary'] if data_text['Summary'].endswith('e') 
else
processed_text = data_text['Summary'].map(preprocess)
processed_text[:10]


# In[30]:


from gensim import corpora, models
dictionary = gensim.corpora.Dictionary(processed_text)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_text]
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
get_lda_topics(lda_model, num_topics)


# In[31]:


train_sentences = [' '.join(text) for text in processed_text]
vectorizer = CountVectorizer(analyzer='word', max_features=5000);
x_counts = vectorizer.fit_transform(train_sentences);
transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
model = NMF(n_components=num_topics, init='nndsvd');
model.fit(xtfidf_norm)


# In[32]:


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


# In[34]:


get_nmf_topics(model, 20)


# In[ ]:




