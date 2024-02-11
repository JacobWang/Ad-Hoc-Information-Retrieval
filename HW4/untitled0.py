# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 02:45:18 2023

@author: Zhengqi Wang
"""

import math
import re
from collections import defaultdict
import stop_list as sl
import numpy as np
from nltk.stem import PorterStemmer

def parser(file_location):
    ps = PorterStemmer()
    hash = {}
    with open(file_location, 'r') as f:
        i = 0
        for line in f:
            if '.I' in line:
                i += 1
            elif '.W' not in line:
                if i not in hash:
                    hash[i]  = list(filter(None, re.split('\W|\d', line.strip()))) 
                else:
                    hash[i] += list(filter(None, re.split('\W|\d', line.strip())))
    f.close()
    for k in hash:
        ls, temp = hash[k], []
        for w in ls:
            if w not in sl.closed_class_stop_words:
                temp.append(ps.stem(w))
        hash[k] = temp
    return hash

def calc_tfidf(hash, length):
    sent_list = list(hash.values())
    idf = {}
    tfidf = defaultdict(lambda: defaultdict(float))
    i = 1
    for sent in sent_list:
        for w in sent:
            if w not in idf.keys():
                count = 0
                for q in sent_list:
                    if w in q:
                        count += 1
                idf[w] = math.log(length / count)
            tf = sent.count(w) / len(sent)
            tfidf[i][w] = tf * idf[w]
        i += 1
    return tfidf

q = parser('cran.qry')
ab = parser('cran.all.1400')
q_tfidf = calc_tfidf(q, 225)
ab_tfidf = calc_tfidf(ab, 1400)

sum_q, sum_ab = 0, 0
cur_q_words, cur_q_tfidf, cur_ab_tfidf = [], [], []
scores = defaultdict(lambda: defaultdict(float))
for i in q_tfidf.keys():
    cur_q_tfidf = list(q_tfidf[i].values())
    cur_q_words = list(q_tfidf[i].keys())
    for ab in ab_tfidf.keys():
        for word in cur_q_words:
            if word not in list(ab_tfidf[ab].keys()):
                cur_ab_tfidf.append(0)
            else:
                cur_ab_tfidf.append(ab_tfidf[ab][word])
        for j in cur_q_tfidf:
            sum_q += j**2
        for k in cur_ab_tfidf:
            sum_ab += k**2
        numerator = np.dot(cur_q_tfidf, cur_ab_tfidf)
        cur_ab_tfidf = []
        denominator = np.sqrt(sum_q * sum_ab)
        sim = 0
        if denominator != 0:
            sim = numerator / denominator
        scores[i][ab] = sim
for i in scores.keys():
    scores[i] = {k: v for k, v in sorted(scores[i].items(), key=lambda item: item[1], reverse=True)} 

with open('output.txt', 'w') as f:
    for x in scores.keys():
        records_written = 0  # Initialize records_written counter
        for y, sim in scores[x].items():
            if records_written < 100:
                f.write(f'{x} {y} {sim:.6f}\n')  # Write with 6 decimal places
                records_written += 1
            elif sim > 0:
                f.write(f'{x} {y} {sim:.6f}\n') 
f.close()

