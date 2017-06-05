# -*- coding: utf-8 -*-

import gensim
import numpy as np
from nltk.corpus import stopwords
from math import sqrt
import _pickle as pickle

stopword = set(stopwords.words('english'))

def cosine_similarity(v1, v2):
    dist = np.dot(v1, v2) / sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    return dist


def get_vector(sentence, wordvec_model, tfidf_model = None):
    
    vec_size = len(wordvec_model.syn0[0])
    sent_vec = np.zeros(vec_size)
    totalweight = 0
    word_count = 0

    if tfidf_model == None:
        for word in [ i for i in sentence.lower().split() if i not in stopword]:
            if word in wordvec_model:
                sent_vec = sent_vec + wordvec_model[word]
                word_count += 1
        if(word_count == 0):
            checkFlag = False
        else:
            checkFlag = True
        sent_vec = sent_vec / word_count
    else:
        for word in [ i for i in sentence.lower().split() if i not in stopword]:
            if word in wordvec_model:
                if word in tfidf_model:
                    sent_vec = sent_vec + wordvec_model[word]*tfidf_model[word]
                    totalweight += tfidf_model[word]
                else:
                    sent_vec = sent_vec + wordvec_model[word]*0
        if(totalweight == 0):
            checkFlag = False
        else:
            checkFlag = True
        sent_vec = sent_vec / totalweight
    return sent_vec, checkFlag

#def get_weighted_vec(sentence, wordvec_model):


if __name__ == '__main__':
    
    if 'model' not in globals():
        model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    
    tfidffile = "D:/Python project/Plagiarism Project/tfidfmodel.pl"
    
    with open(tfidffile, 'rb') as tfidf_file:
        tfidf_model = dict(pickle.load(tfidf_file))
        
    sentence1 = "Subject to agent great hours of operation and availability ."
    sentence2 = "Subject to agent many hours of operation and availability ."
    
    s1 = get_vector(sentence1, model, tfidf_model)
    s2 = get_vector(sentence2, model, tfidf_model)
    
    print(cosine_similarity(s1, s2))
    
    print("pass")