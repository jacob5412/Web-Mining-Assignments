#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:49:00 2018

@author: jacobjohn

1) Write a program to do stop word removal and stemming from a paragraph.
2) Prepare a table that includes “Word” and “frequency (2 columns). 
3) Print the frequency of words (terms) start with A,B,S,D,E. 
4) Find maximum frequency terms/term.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import pandas as pd
import re
from nltk.corpus import inaugural

##Using Obama's inaugural speech
Obama = inaugural.words(fileids = '2009-Obama.txt')

##class to print headers in color
class color:
   BOLD = '\033[1m'
   END = '\033[0m'

example_sent = Obama
 
##stopword removal
stop_words = set(stopwords.words('english'))
 
filtered_sentence = [w for w in Obama if not w in stop_words]

##stemming with porter and snowball
processed = []
stemmer = SnowballStemmer("english")
processed = [stemmer.stem(i) for i in filtered_sentence]

processed2 = []
ps = PorterStemmer()
processed2 = [ps.stem(i) for i in filtered_sentence]
 
##Plotting distribution of tokens
fd = nltk.FreqDist(Obama)
fd.plot(50,cumulative=False)

print("\nFiltered sentence after stopword removal is (first 5): ",filtered_sentence[1:5])
print("\nWords after Snowball stemming (first 25): ",processed[1:25])
print("\nWords after Porter stemming (first 25): ",processed2[1:25])

#declare a dictionary for frequency
word_freq = {}
for tok in filtered_sentence:
    if tok in word_freq:
        word_freq[tok] += 1
    else:
        word_freq[tok] = 1
        
print("\n{:<15} {:<5}".format(color.BOLD + 'Word','Frequency' + color.END))
for key in word_freq:
    print("{:<15} {:<5}".format(key,word_freq[key]))

##printing out words starting with A,B,S,D,E
starts_with = {}

for key in word_freq:
    if key[0] in 'ABSDE':
        starts_with[key] = word_freq[key]

print("\nFrequency of words (terms) start with A,B,S,D,E are: ")
print("{:<15} {:<5}".format(color.BOLD + 'Word','Frequency' + color.END))
for key in starts_with:
    print("{:<15} {:<5}".format(key,starts_with[key]))

##Printing out most frequent words
max_dict = {}
while len(max_dict) < 5:
    max_val = 0
    for key in word_freq:
        if max_val < word_freq[key] and re.match(r'[A-Za-z]+',key) and key not in max_dict:
            max_key = key
            max_val = word_freq[key]
    max_dict[max_key] = max_val
print("\nThe most frequent words are: ")
print("{:<15} {:<5}".format(color.BOLD + 'Word','Frequency' + color.END))
for key in max_dict:
    print("{:<15} {:<5}".format(key,max_dict[key]))
    
##Exporting frequency table into excel
df = pd.DataFrame(list(word_freq.items()),columns=['Word','Frequency'])
df.to_excel('test.xlsx', sheet_name='sheet1', index=True)