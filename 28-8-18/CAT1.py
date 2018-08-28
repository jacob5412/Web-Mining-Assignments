#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:36:37 2018

@author: jacobjohn

a) Take dynamic input from user as a paragraph (give input > 20 lines) and remove
punctuations first (only). Then print resulting paragraph (after removing punctuations).
b) Afterwards, remove stop words and print the whole paragraph again.
c) Next, find the frequencies of those words that start with vowel and write the result in a
excel file with two columns of “words” and “freq”.
d) Prepare a dictionary that shall contain all words (after removing punctuations and stop
words) and apply stemming to reduce inflected words to their word stem, base
or root form. 
Search for any word and print the details

"""
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import pandas as pd

string = str()

##entering 20 lines
while(True):
    sent = nltk.sent_tokenize(string)
    if(len(sent) > 20):
        break
    else:
        print("Enter paragraph over 20 sentences long")
        string = str(input("Enter the paragraph: "))
        
# define punctuation
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

# remove punctuation from the string
no_punct = ""
for char in string:
   if char not in punctuations:
       no_punct = no_punct + char

# display the unpunctuated string
print("Paragraph without punctuation: ",no_punct)

##stopword removal
stop_words = set(stopwords.words('english')) 
filtered_sentence = [w for w in sent if not w in stop_words]
print("After stopword removal: ", sent)

##stopword removal of non punctuation
no_punct = nltk.word_tokenize(no_punct)
filtered_sentence2 = [w for w in no_punct if not w in stop_words]

##word tokenize
word = nltk.word_tokenize(string)

##count vowel frequency
word_freq = {}
for tok in word:
    if tok in word_freq:
        word_freq[tok] += 1
    elif tok[0] in 'AEIOUaeiou':
        word_freq[tok] = 1
        
##counting word frequency
word_freq2 = {}
for tok in word:
    if tok in word_freq2:
        word_freq2[tok] += 1
    else:
        word_freq2[tok] = 1
        
##Exporting frequency table into excel
df = pd.DataFrame(list(word_freq.items()),columns=['Word','Frequency'])
df.to_excel('test.xlsx', sheet_name='sheet1', index=True)

##stemming with porter and snowball
processed = {}
stemmer = SnowballStemmer("english")
for i in filtered_sentence2:
    processed[i] = stemmer.stem(i)

processed2 = {}
ps = PorterStemmer()
for i in filtered_sentence2:
    processed2[i] = ps.stem(i)


while(True):
    print("Enter a word to search or leave blank to exit")
    user = input()
    if user:
        print("Details are as follows: ")
        print("Porter Stemming: ",processed2[user] if user in processed2 else 'False')
        print("Snowball Stemming: ",processed[user] if user in processed else 'False')
        print("Frequency: ",word_freq2[user] if user in word_freq2 else 'False')
        print("Vowel status: ",word_freq[user] if user in word_freq else 'False')
    else:
        break