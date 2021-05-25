#!/usr/bin/env python
# coding: utf-8

# In[69]:


import cv2
import pytesseract
from pytesseract import Output
import os
import numpy as np
import re
import pandas as pd


# In[70]:


def find_dates(d, n_boxes, img, date, date_valid):
    dates = ['^(0[1-9]|1[012])[/-](0[1-9]|[12][0-9]|3[01])[/-](19|20)\d\d$', '^(0[1-9]|[12][0-9]|3[01])[/-](0[1-9]|1[012])[/-](19|20)\d\d$', '^(19|20)\d\d[/-](0[1-9]|1[012])[/-](0[1-9]|[12][0-9]|3[01])$' ]
    temp = '(?:% s)' % '|'.join(dates)
    for i in range(n_boxes):
        text = d['text'][i]
        if re.findall(temp, d['text'][i]):
            text = text.replace('-', '/')
            if bool(re.search('^(0[1-9]|[12][0-9]|3[01])[/-](0[1-9]|1[012])[/-](19|20)\d\d$', text)):
                text = re.sub(r'(\d+)[/-](\d+)[/-](\d+)', r'\2/\1/\3', text)
            elif bool(re.search('^(19|20)\d\d[/-](0[1-9]|1[012])[/-](0[1-9]|[12][0-9]|3[01])$', text)):
                text = re.sub(r'(\d+)[/-](\d+)[/-](\d+)', r'\2/\3/\1', text)
            result = text == date
            date_valid.append(result)
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(directory + '\checked ' + receipt, img)


# In[72]:


def find_keywords(d, n_boxes, img, keyword_valid, receipt):
    keywords = ['COVID', 'Sputnik', 'AstraZeneca', 'Sinotech', 'Pfizer']
    for i in range(n_boxes):
        text = d['text'][i]
        if text in keywords:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            keyword_valid.append(receipt)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(directory + '\checked ' + receipt, img)


# In[73]:


directory = r'C:\Users\chloe\OneDrive\Desktop\Receipts\Edited-Both'

df = pd.read_csv(r'C:\Users\chloe\OneDrive\Desktop\OCR Values.csv')
df.head(5)

date = df.iloc[0,0]
date_valid = []
keyword_valid = []

for receipt in df['Files']:
    img = cv2.imread(directory + '\\' + receipt)
    
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #Adjusts contrast and brightness
    brightness = 60
    contrast = 60
    img = np.int16(img)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    
    
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    img=cv2.divide(img, bg, scale=255)
 
    thres, img=cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    
    
    #calls correct function when uncommented
    find_dates(d, n_boxes, img, date, date_valid)
    find_keywords(d, n_boxes, img, keyword_valid, receipt)
    
print(date_valid)
print(keyword_valid)


# In[ ]:





# In[ ]:




