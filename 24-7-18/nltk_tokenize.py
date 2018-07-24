#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:40:27 2018

@author: jacobjohn

Write a program using nltk toolkit to tokenize:
    a) Sentence
    b) Multiple sentences
    c) A paragraph
    d) Information of a complete web page
"""
import nltk

##Tokenizing a sentence
from nltk.tokenize import word_tokenize
string_input = str(input("Enter a sentence: "))
tokens = word_tokenize(string_input)
print("The tokens for this sentence are: ",tokens)

##Tokenizing mulitple sentences
print("Enter multiple sentences: ")
lines = []
while True:
    line = input()
    if line:
        lines.append(line)
    else:
        break
tok = []
for t in lines:
    t = word_tokenize(t)
    tok.append(t)
print("Tokens for multiple sentences are as follows: ",tok)

##Tokenizing paragraphs
text = input("Enter a paragraph: ")
paragraph = nltk.sent_tokenize(text) # this gives us a list of sentences
# now loop over each sentence and tokenize it separately
for sentence in paragraph:
    tokenized_text = nltk.word_tokenize(sentence)
    print(tokenized_text)
    
##Information of a complete web page
#bs4 module to crawl webpage
from bs4 import BeautifulSoup
import urllib.request 
#requesting php.net for information
response = urllib.request.urlopen('http://php.net/') 
html = response.read()
#cleaning grabbed text
soup = BeautifulSoup(html,"html5lib")
text = soup.get_text(strip=True)
tokens_web = word_tokenize(text)
print("Tokens for this web page are: ",tokens_web[1:15])
#declare a dictionary
word_freq = {}
for tok in tokens_web:
    tok = tok.split()
    for t in tok:
        if t in word_freq:
            word_freq[t] += 1
        else:
            word_freq[t] = 1
#Frequency of top five words
import re
max_dict = {}
while len(max_dict) < 5:
    max_val = 0
    for key in word_freq:
        if max_val < word_freq[key] and re.match(r'[A-Za-z]+',key) and key not in max_dict:
            max_key = key
            max_val = word_freq[key]
    max_dict[max_key] = max_val
print("\nThe five most frequent words are: ")
for key in max_dict:
    print(key,":",max_dict[key])