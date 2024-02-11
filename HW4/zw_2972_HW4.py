# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:33:37 2023
Ad Hoc Information Retrieval task using TF-IDF weights and cosine similarity scores
@author: Zhengqi Wang
"""
import nltk
import string
import numpy as np
import math
from collections import defaultdict
from nltk.stem import PorterStemmer
import sys
if len(sys.argv) != 4:
    print("Usage: python zw_2972_HW4.py qureies_file abstracts_file output_file")
    sys.exit(1)
    
nltk.download('stopwords')
nltk.download('punkt')

query_file = sys.argv[1]
abstracts_file = sys.argv[2]
output_file = sys.argv[3]

closed_class_stop_words = ['a', 'the', 'an', 'and', 'or', 'but', 'about', 'above', 'after', 'along', 'amid', 'among',
                           'as', 'at', 'by', 'for', 'from', 'in', 'into', 'like', 'minus', 'near', 'of', 'off', 'on',
                           'onto', 'out', 'over', 'past', 'per', 'plus', 'since', 'till', 'to', 'under', 'until', 'up',
                           'via', 'vs', 'with', 'that', 'can', 'cannot', 'could', 'may', 'might', 'must',
                           'need', 'ought', 'shall', 'should', 'will', 'would', 'have', 'had', 'has', 'having', 'be',
                           'is', 'am', 'are', 'was', 'were', 'being', 'been', 'get', 'gets', 'got', 'gotten',
                           'getting', 'seem', 'seeming', 'seems', 'seemed',
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these',
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace',
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their', 'theirs',
                           'you', 'your', 'yours', 'me', 'my', 'mine', 'I', 'we', 'us', 'much', 'and/or'
                           ]

stopwords = set([
    *nltk.corpus.stopwords.words('english'),
    *closed_class_stop_words
])

stemmer = nltk.stem.snowball.EnglishStemmer()


# Load queries
queries = {}
with open(query_file, "r") as file:
    lines = file.readlines()
    query_id = None
    query_text = ""
    for line in lines:
        if line.startswith(".I"):
            if query_id is not None:
                queries[query_id] = query_text
            query_id = int(line.strip().split()[-1])
            query_text = ""
        else:
            query_text += line
    if query_id is not None:
        queries[query_id] = query_text

# Load abstracts
abstracts = {}
with open(abstracts_file, "r") as file:
    lines = file.readlines()
    abstract_id = None
    abstract_text = ""
    is_in_abstract = False  # Add a flag to indicate if we are inside an abstract
    skip_lines_after_A_or_B = False  # Add a flag to skip lines after .A or .B
    for line in lines:
        if line.startswith(".I"):
            if abstract_id is not None:
                abstracts[abstract_id] = abstract_text
            abstract_id = int(line.strip().split()[-1])
            abstract_text = ""
            is_in_abstract = False  # Reset the flag for a new abstract
        elif line.startswith(".T") or line.startswith(".W") or line.startswith(".A") or line.startswith(".B"):
            is_in_abstract = True  # Start capturing text for the abstract
            # Add a space for new lines following .T or .W
            if abstract_text and not abstract_text.endswith(" "):
                abstract_text += ""
        elif is_in_abstract:
            abstract_text += line

    # Handle the last abstract
    if abstract_id is not None:
        abstracts[abstract_id] = abstract_text




# Tokenization and preprocessing functions

ps = PorterStemmer()
def tokenize_and_preprocess(text):
    # Tokenize the text
    tokens = nltk.tokenize.word_tokenize(text)
    
    # Remove punctuation and numbers, and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in string.punctuation]
    
    # Remove stop words and apply stemming
    stemmed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stopwords]
    
    return stemmed_tokens  # Return the stemmed tokens

queries_preprocessed = {query_id: tokenize_and_preprocess(query_text) for query_id, query_text in queries.items()}
abstracts_preprocessed = {abstract_id: tokenize_and_preprocess(abstract_text) for abstract_id, abstract_text in abstracts.items()}


total_abstracts_len = len(abstracts)
total_queries_len = len(queries_preprocessed)
word_frequencies = {}


# Calculate TF-IDF for queries
qureis_tfidf = {}
sent_list = list(queries_preprocessed.values())
q_idf = {}
i = 1

for sent in sent_list:
    qureis_tfidf[i] = {}
    for w in sent:
        if w not in q_idf.keys():
            count = 0
            for q in sent_list:
                if w in q:
                    count += 1
            q_idf[w] = math.log(len(queries_preprocessed) / (count + 1))
        tf = sent.count(w) / len(sent)
        qureis_tfidf[i][w] = tf * q_idf[w]
    i += 1


abstracts_tfidf = {}
sent_list = list(abstracts_preprocessed.values())
a_idf = {}
i = 1
for sent in sent_list:
    abstracts_tfidf[i] = {}
    for w in sent:
        if w not in a_idf.keys():
            count = 0
            for q in sent_list:
                if w in q:
                    count += 1
            a_idf[w] = math.log(len(abstracts_preprocessed) / (count + 1))
        tf = sent.count(w) / len(sent)
        abstracts_tfidf[i][w] = tf * a_idf[w]
    i += 1


sum_quries, sum_abstracts = 0, 0
current_queries_words, current_queries_tfidf, current_abstracts_tfidf = [], [], []
scores = defaultdict(lambda: defaultdict(float))
for i in qureis_tfidf.keys():
    current_queries_words = list(qureis_tfidf[i].keys())
    current_queries_tfidf = list(qureis_tfidf[i].values())
    
    for a in abstracts_tfidf.keys():
        current_abstracts_tfidf = []  
        for word in current_queries_words:
            if word not in list(abstracts_tfidf[a].keys()):
               current_abstracts_tfidf.append(0)
            else:
                current_abstracts_tfidf.append(abstracts_tfidf[a][word])
        for j in current_queries_tfidf:
           sum_quries += j**2
        for k in current_abstracts_tfidf:
            sum_abstracts += k**2
        numerator = np.dot(current_queries_tfidf, current_abstracts_tfidf)
        cur_ab_tfidf = []
        denominator = np.sqrt(sum_quries * sum_abstracts)
        similarities = 0
        if denominator != 0:
            similarities = numerator / denominator
        scores[i][a] =  similarities

for i in scores.keys():
    scores[i] = {k: v for k, v in sorted(scores[i].items(), key=lambda item: item[1], reverse=True)} 

with open(output_file, 'w') as f:
    for x in scores.keys():
        records_written = 0  # Initialize records_written counter
        for y, sim in scores[x].items():
            if records_written < 100:
                f.write(f'{x} {y} {sim:.6f}\n')  # Write with 6 decimal places
                records_written += 1
            elif sim > 0:
                f.write(f'{x} {y} {sim:.6f}\n') 
f.close()
        
        
         