

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:10:43 2018

@author: jacobjohn

1. Write a program to tokenize
 a) A sentence
 b) Mutliple sentences
"""

#Tokenize an individual sentence
string_sentence = str(input("Enter a sentence: "))
string_tok = string_sentence.split()
print("Tokens are as follows: ",string_tok)

#Tokenize mulitple sentences
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
    tok.append(t.split())
print("Tokens for multiple sentences are as follows: ",tok)
```

    Enter a sentence: This is a sentence.
    Tokens are as follows:  ['This', 'is', 'a', 'sentence.']
    Enter multiple sentences: 
    This is 1.
    This is another.
    This is three.
    
    Tokens for multiple sentences are as follows:  [['This', 'is', '1.'], ['This', 'is', 'another.'], ['This', 'is', 'three.']]



```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:34:25 2018

@author: jacobjohn

2. Write a program to count the frequency of tokenzied words from two sentences.
"""
#input two sentences
print("Enter two sentences: ")
lines = []
i = 0
while(i < 2):
    line = input()
    lines.append(line)
    i += 1

#declare a dictionary
word_freq = {}
for tok in lines:
    tok = tok.split()
    for t in tok:
        if t in word_freq:
            word_freq[t] += 1
        else:
            word_freq[t] = 1
            
print("Frequency distribution of words are: ",word_freq)
```

    Enter two sentences: 
    This is one sentence.
    This is also one sentence.
    Frequency distribution of words are:  {'This': 2, 'is': 2, 'one': 2, 'sentence.': 2, 'also': 1}



```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:40:27 2018

@author: jacobjohn

3. Write a program using nltk toolkit to tokenize:
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
```

    Enter a sentence: This is a sentence.
    The tokens for this sentence are:  ['This', 'is', 'a', 'sentence', '.']
    Enter multiple sentences: 
    These are sentences.
    Over the rainbow.
    
    Tokens for multiple sentences are as follows:  [['These', 'are', 'sentences', '.'], ['Over', 'the', 'rainbow', '.']]
    Enter a paragraph: This is a sentence. So is this one. I hope this works.
    ['This', 'is', 'a', 'sentence', '.']
    ['So', 'is', 'this', 'one', '.']
    ['I', 'hope', 'this', 'works', '.']
    Tokens for this web page are:  [':', 'Hypertext', 'PreprocessorDownloadsDocumentationGet', 'InvolvedHelpGetting', 'StartedIntroductionA', 'simple', 'tutorialLanguage', 'ReferenceBasic', 'syntaxTypesVariablesConstantsExpressionsOperatorsControl', 'StructuresFunctionsClasses', 'and', 'ObjectsNamespacesErrorsExceptionsGeneratorsReferences', 'ExplainedPredefined', 'VariablesPredefined']
    
    The five most frequent words are: 
    PHP : 79
    of : 67
    the : 62
    and : 50
    can : 42



```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:41:41 2018

@author: jacobjohn

4. Write a program to remove the stop words stemming using nltk toolkit.
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

```

    Enter a paragraph: his is a sentence. So is this one. I hope this works.
    Tokens are:  ['his', 'is', 'a', 'sentence', '.', 'So', 'is', 'this', 'one', '.', 'I', 'hope', 'this', 'works', '.']
    
    Filtered sentence after stopword removal is:  ['sentence', '.', 'So', 'one', '.', 'I', 'hope', 'works', '.']
    
    Words after Snowball stemming:  [['sentenc', '.', 'so', 'one', '.', 'i', 'hope', 'work', '.']]
    
    Words after Porter stemming:  [['sentenc', '.', 'So', 'one', '.', 'I', 'hope', 'work', '.']]

