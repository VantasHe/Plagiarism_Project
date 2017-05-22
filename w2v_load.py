# -*- coding: utf-8 -*-

import gensim
import numpy as np
from nltk.corpus import stopwords
from math import sqrt

stopword = set(stopwords.words('english'))

def cosine_similarity(v1, v2):
    dist = np.dot(v1, v2) / sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    return dist


def get_vector(sentence, wordvec_model):
    
    vec_size = len(wordvec_model.syn0[0])
    sent_vec = np.zeros(vec_size)
    word_count = 0
    for word in [ i for i in sentence.lower().split() if i not in stopword]:
        if word in model:
            sent_vec = sent_vec + model[word]
            word_count += 1
    sent_vec = sent_vec / word_count
    return sent_vec


if __name__ == '__main__':
    
    if 'model' not in globals():
        model = gensim.models.KeyedVectors.load_word2vec_format("../word2vec_project/GoogleNews-vectors-negative300.bin", binary=True)
        
    sentence1 = "A break-in at the U.S. Justice Department 's World Wide Web site last week highlighted the Internet 's continued vulnerability to hackers ."
    sentence2 = "Fidelity officials immediately closed the loophole identified by the magazine , a spokeswoman said . But multiple security measures previously in place would have prevented a security breach despite the hole , the spokeswoman added ."
    
    s1 = get_vector(sentence1, model)
    s2 = get_vector(sentence2, model)
    
    print(cosine_similarity(s1, s2))
    
    print("pass")