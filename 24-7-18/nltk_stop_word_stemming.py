#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:41:41 2018

@author: jacobjohn

Write a program to remove the stop words stemming using nltk toolkit.
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer

example_sent = input("Enter a paragraph: ")
 
stop_words = set(stopwords.words('english'))
 
word_tokens = word_tokenize(example_sent)
 
filtered_sentence = [w for w in word_tokens if not w in stop_words]
 
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

processed = []
stemmer = SnowballStemmer("english")
intermediate = [stemmer.stem(i) for i in filtered_sentence]
processed.append(intermediate)

processed2 = []
ps = PorterStemmer()
intermediate = [ps.stem(i) for i in filtered_sentence]
processed2.append(intermediate)
 
print("Tokens are: ",word_tokens)
print("\nFiltered sentence after stopword removal is: ",filtered_sentence)
print("\nWords after Snowball stemming: ",processed)
print("\nWords after Porter stemming: ",processed2)
