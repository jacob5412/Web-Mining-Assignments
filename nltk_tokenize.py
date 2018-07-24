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

#Tokenizing a sentence
from nltk.tokenize import word_tokenize
string_input = str(input("Enter a sentence: "))
tokens = word_tokenize(string_input)
print("The tokens for this sentence are: ",tokens)

#Tokenizing mulitple sentences
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

#Tokenizing paragraphs
text = input("Enter a paragraph: ")
paragraph = nltk.sent_tokenize(text) # this gives us a list of sentences
# now loop over each sentence and tokenize it separately
for sentence in paragraph:
    tokenized_text = nltk.word_tokenize(sentence)
    print(tokenized_text)
    
#Information of a complete web page
