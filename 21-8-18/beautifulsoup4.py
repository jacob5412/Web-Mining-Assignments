#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:44:57 2018

@author: jacobjohn

Source: https://notawful.org/basic-python-web-crawler/
"""

#scrapping the web using BeautifulSoup

import requests
from bs4 import BeautifulSoup
def web(page,WebUrl):
    if(page>0):
        url = WebUrl
        code = requests.get(url)
        plain = code.text
        s = BeautifulSoup(plain, "html.parser")
        for link in s.findAll('a',{'class':'post-card-content-link'}):
            tet = link.get('href')
            print("href " + tet)
            url_low = WebUrl + tet
            code_low = requests.get(url_low)
            plain_low = code_low.text
            s_low = BeautifulSoup(plain_low, "html.parser")
            for img in s_low.findAll('img'):
                tet_low = img.get('src')
                print(tet_low)

web(1,'https://notawful.org')