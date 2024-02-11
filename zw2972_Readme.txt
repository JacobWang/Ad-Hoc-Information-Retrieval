Jacob Wang(zw2972) Ad Hoc Information Retrieval task using TF-IDF weights and cosine similarity scores
Oct 10th 2023 
To run the .py file use
    
    python zw_2972_HW4.py <queries_file> <abstracts_file> <output_file>
    
The code is bascally first reading the queries_file and the abstracts file and stored them 
    by the queriy id and abstract id in to a dictionary
    
Then we tokenize the word, and removing suffix, and remove punctuation, remove the word that is in the stop word list.

Then calculate the tf-idf value for both the abstarcts and the quries.

Finally, we use the tf-idf score to calculate the similarity scores. 

The script calculates TF-IDF scores for queries and documents and ranks documents based on their similarity to queries. 
The retrieval results are written to the specified output_file in the following format:
            <QueryID> <DocumentID> <SimilarityScore>
            <QueryID> <DocumentID> <SimilarityScore>
            ...
For the output, we only want the one that is bigger than 0, if the 100th element is equal to 0 by order, then we stop adding it to the file