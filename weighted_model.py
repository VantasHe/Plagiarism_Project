# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:09:12 2017

@author: Falcon4
"""

import os
import re
import _pickle as pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import mytools


class model:
    
    def __init__(self, dm_root_path = None, tfidf_model_path = None):
        self.path = dm_root_path
        self.file_list = []
        if self.path != None:
            self.file_list = mytools.get_walkthrought_dir(self.path)
            self.sFilePath = self.path + '../_tfidffile'
        self.file_count = 0
        self.tfidf_model = dict()
        if tfidf_model_path !=None:
            self.tfidf_model = self.load(tfidf_model_path)
            
    def save(self, save_path):
        if not os.path.exists(save_path): 
            os.mkdir(save_path)
        with open(save_path, "wb") as save_model:
            pickle.dump(self.tfidf_model, save_model)
            
    
    def do_tfidf(self):
        corpus = []
        for ff in self.file_list:
            if re.search('.txt', ff[2]):
                with open(ff[0],'r+', encoding='utf8') as f:
                    content = f.read()
                    self.file_count += 1
                corpus.append(content)
        print("total load {0} files and {1} corpus".format(self.file_count, len(corpus)))
        
        """ Calculate tf(term frequency) and df(document term)"""
        term_fq_vectorizer = CountVectorizer()      # Instance to count all terms in each document 
        tfidfizer = TfidfTransformer()
        term_fq = term_fq_vectorizer.fit_transform(corpus).toarray()  # Transform tf into array
        weight = tfidfizer.fit_transform(term_fq).toarray()
        word = term_fq_vectorizer.get_feature_names() # Get keywords from all the corpus
        sum_weights = weight.sum(axis=0).tolist()
        
        """ Sort terms by weight """
        self.tfidf_model = dict(zip(word, sum_weights))     # Bind term name with its weight and transform to dictionary.
        
        """ Write to file """
        self.save(self.sFilePath+"/tfidfmodel.pl")
            
        """
        # Store Tf-idf results of whole documet to './tfidffile'
        for i in range(len(weight)) :
            print("--------Writing all the tf-idf in the", i, " file into ", sFilePath + '/' + str(i).zfill(5) + '.txt', "--------")
            f = open(sFilePath+'/'+ str(i).zfill(5) +'.txt','w+')
            for j in range(len(word)) :
                f.write(word[j]+"    "+str(weight[i][j])+"\n")
            f.close()
        """
        return self.tfidf_model

    
    def get_sorted_list(self, top_n = None, save_file = False):
        sorted_list = sorted(self.tfidf_model.items(),
                             key=lambda d: d[1], reverse=True)    # Sort descending by value.
        
        if top_n != None:
            sort_list = dict(sorted_list[:top_n])
        else:
            sort_list = self.tfidf_model

            
        if save_file == False:
            return sort_list
        elif save_file == True:
            print("--------Writing all the tf-idf in the Database file into ",
                  self.sFilePath + '/' + 'TotalBase.txt', "--------")
            with open(self.sFilePath + '/TotalBase.txt', 'w+', encoding='utf8') as f:
                for pair in sort_list :
                    f.write( str(pair[0]) + "\t" + str(pair[1]) + "\n" )
            return sort_list

        
    @classmethod
    def load(self, model_path):
        self.sFilePath = model_path
        with open(model_path, 'rb') as tfidf_file:
            self.tfidf_model = dict(pickle.load(tfidf_file))
        return self.tfidf_model