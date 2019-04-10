#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[8]:


supplier_text = pd.read_csv('distinct_supplier_text.csv')


# In[9]:


supplier_text.head()


# In[10]:


# Cleaning Text Data
import string
import nltk
import re

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[11]:


# import nltk and download punctuations
nltk.download('punkt')


# In[12]:


supplier_text.shape


# In[13]:


supplier_text.columns


# In[14]:


# Take the relevant text of the suppliers data
supplier1 = supplier_text['home_page_text']
supplier2 = supplier_text['about_or_contact_text']


# In[15]:


# Join text together to process to cleaning
supplier_text_join = pd.DataFrame(supplier1 + supplier2)


# In[16]:


supplier_text_join = supplier_text_join.rename(columns={0:"supplier_text"})


# In[17]:


supplier_text_join['supplier_name'] = supplier_text['company_name']


# In[18]:


supplier_text_join.head()


# In[22]:


# Remove all the '\n' line-brakes
supplier_text_join.supplier_text = supplier_text_join.supplier_text.str.replace(r'\n',' ')


# In[23]:


# Remove 'non-ascii' characters
supplier_text_join.supplier_text = supplier_text_join.supplier_text.str.replace(r'[^\x00-\x7F]',' ')
# Make all the text lowercase
supplier_text_join.supplier_text = supplier_text_join.supplier_text.str.lower()


# In[24]:


# Remove postcode
supplier_text_join.supplier_text = supplier_text_join.supplier_text.str.replace(r'[a-z]{1,2}[0-9r][0-9a-z]? [0-9][a-z]{2}','')


# In[25]:


# Remove numbers 
supplier_text_join.supplier_text = supplier_text_join.supplier_text.str.replace(r'\d+','')


# In[26]:


# Remove punctuations
supplier_text_join['clean_text'] = supplier_text_join.supplier_text.apply(
    lambda x:str(x).translate(str.maketrans('','', string.punctuation)))


# In[27]:


# Tokenise all of the clean text
supplier_text_join['token_text'] = supplier_text_join.clean_text.apply(word_tokenize)
                                       


# In[28]:


supplier_text_join.head()


# In[29]:


# Save the clean text for supplier in a csv file!
supplier_text_join.to_csv('supplier_text_clean.csv')

