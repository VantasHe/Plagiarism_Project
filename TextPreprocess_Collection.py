# Author : Vick
# Date   : 05/26/2017
# Version: 1.00
import os
import sys
import re
import string
import time

import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import _pickle as pickle

import mytools

cachedStopWords = stopwords.words("english")

def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)

    
""" Method of preprocessing file , like stemming. """
def preprocessFile(file_info, dm_root_path, stem_mode = False) :
    
    fsource = file_info[0]
    dm_path = file_info[1]
    fname = file_info[2]
    
    sFilePath = dm_root_path+'/_segfile'   # Create the directory to store segments of files.
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)
             
    with open(fsource,'r+', encoding='utf8') as f:   # Open file, mode: read only.
        fcontext = f.read()            # Read file content and store as string.

    
    """ Remove punctuation from document """
    deletetable = str.maketrans({key:" " for key in string.punctuation})    # Make translation table of punctuation.
    text = fcontext.translate(deletetable)     # Remove punctuation from file.
    #text = ' '.join([word for word in fcontext.split() if word not in cachedStopWords])
    """ Turn document into lowercase and remove Stopwords """
    text = ' '.join([word for word in text.lower().split() if word not in cachedStopWords])
    seg_list = text.split() 
    #seg_list = fcontext.split()
    
    """ Method : stemming """
    if stem_mode == True :
        st = LancasterStemmer()
        result = []
        for seg in seg_list :
            st_seg = st.stem(seg)
            if(st_seg != '' and st_seg != '\n'):
                result.append(st_seg)
    elif stem_mode == False:
        result = seg_list

    """
    wnl= WordNetLemmatizer()
    for seg in seg_list :
        wnl_seg = wnl.lemmatize(seg)
        if(wnl_seg != '' and wnl_seg != '\n'):
            result.append(wnl_seg)
    """

    """ Write into File """
    destPath = re.sub(dm_root_path, sFilePath, dm_path)
    finalname = re.sub('.txt', "-seg.txt", fname)
    with open(destPath+finalname, "w+", encoding='utf8') as seg_file:
        seg_file.write(' '.join(result))


""" Calculate Tf-idf and the weight of each term """
def Tfidf(file_info, dm_root_path):
    path = dm_root_path+'/_segfile/'   # Load index of directory of preprocessing datasets.
    filelist = mytools.get_walkthrought_dir(path)

    """ Load all corpus"""
    corpus = []  # Store the result of the doucment feature
    count = 0
    for ff in filelist:
        if re.search('.txt', ff[2]):
            with open(ff[0],'r+', encoding='utf8') as f:
                content = f.read()
                count += 1
            corpus.append(content)

    print("total load {0} files and {1} corpus".format(count, len(corpus)))

    """
    vectorizer = CountVectorizer()    
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names() # Get keywords from all the corpus
    weight = tfidf.toarray()              # Imply the Matrix of Tf-idf
    """

    """ Calculate tf(term frequency) and df(document term)"""
    term_fq_vectorizer = CountVectorizer()      # Instance to count all terms in each document 
    tfidfizer = TfidfTransformer()
    term_fq = term_fq_vectorizer.fit_transform(corpus).toarray()  # Transform tf into array
    weight = tfidfizer.fit_transform(term_fq).toarray()
    word = term_fq_vectorizer.get_feature_names() # Get keywords from all the corpus
    sum_weights = weight.sum(axis=0).tolist()
    
    """ Sort terms by weight """
    dictionary = dict(zip(word, sum_weights))     # Bind term name with its weight and transform to dictionary.
    sort_dict = sorted(dictionary.items(), key=lambda d: d[1], reverse=True)    # Sort descending by value. 
    
    """ Write to file """
    sFilePath = dm_root_path + '/_tfidffile'
    if not os.path.exists(sFilePath): 
        os.mkdir(sFilePath)
        
    """
    # Store Tf-idf results of whole documet to './tfidffile'
    for i in range(len(weight)) :
        print("--------Writing all the tf-idf in the", i, " file into ", sFilePath + '/' + str(i).zfill(5) + '.txt', "--------")
        f = open(sFilePath+'/'+ str(i).zfill(5) +'.txt','w+')
        for j in range(len(word)) :
            f.write(word[j]+"    "+str(weight[i][j])+"\n")
        f.close()
    """

    print("--------Writing all the tf-idf in the Database file into ", sFilePath + '/' + 'TotalBase.txt', "--------")
    with open(sFilePath + '/TotalBase.txt', 'w+', encoding='utf8') as f:
        for pair in sort_dict :
            f.write( str(pair[0]) + "\t" + str(pair[1]) + "\n" )

    with open(sFilePath+"/tfidfmodel.pl", "wb") as save_model:
        pickle.dump(sort_dict, save_model)
        
if __name__ == "__main__":
    argv = "/Users/vick/Documents/Python/Training_Data/Html_CityU2_adjust"
    dm_dir_info = mytools.get_walkthrought_dir(argv)
    
#    timeStart = time.time()
#    for index, ff in enumerate(dm_dir_info):
#        print("Preprocessing on " + ff[2])
#        preprocessFile(ff, argv)
#    timeEnd = time.time()
#    print("Total preprocessing time : {0:f} sec".format(timeEnd - timeStart))
    
    print("Tf-idf start:")
    timeStart = time.time()
    Tfidf(dm_dir_info, argv)
    timeEnd = time.time()
    print("Tf-idf time : {0:f} sec".format(timeEnd - timeStart))