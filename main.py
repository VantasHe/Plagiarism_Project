
import gensim
import re
import time
import os.path
import _pickle as pickle

import mytools
import w2v_load
import weighted_model

if __name__ == '__main__':
    
    argv = r"D:/dataset/HTML_CityU3_adjust"
    dm_dir_info = mytools.get_walkthrought_dir(argv)
    print("====Walk through {path}: total {dm}====".format(path=argv, dm=len(dm_dir_info)))
    
    print("Load Word2Vec model:")
    if 'model' not in globals():
        model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    print("====Word2Vec model loaded!====")
    
    
    print("Load Tfidf model:")
    if 'weights_model' not in globals():
        tfidffile = "D:/Python project/Plagiarism Project/tfidfmodel2.pl"
        """
        with open(tfidffile, 'rb') as tfidf_file:
            weights_model = dict(pickle.load(tfidf_file))
        """
        if os.path.exists(tfidffile):
            print("----Load existed model----")
            tfidf_model = weighted_model.model(dm_root_path = None, tfidf_model_path = tfidffile)
            weights_model = tfidf_model.get_sorted_list(top_n = 500)
        else:
            print("----Training Tfidf model----")
            tfidf_model = weighted_model.model(argv).do_tfidf()
        
            
    print("====Tfidf model loaded!====")
    
#    tfidf_model = None
    
    print("Start to train Sentence Vector:")
    if 'dm_all_vec' not in globals():
        print("====Start to train vector.====")
        starttime = time.time()
        dm_vec = []
        dm_name = []
        for ff in dm_dir_info:
            with open(ff[0], 'r', encoding='utf-8') as dm:
                content = dm.read()
            vec_temp, checkflag = w2v_load.get_vector(content, model, weights_model)
            if checkflag == False:
                print("Failed :" + ff[0])
                dm_dir_info.remove(ff)
            else:
                dname = re.sub(argv+r'\\', r'', str(ff[1]))
                fname = re.sub(r'.txt', r'', str(ff[2]))
                dm_name.append(dname[:2]+'_'+fname)
                dm_vec.append(vec_temp)
        dm_all_vec = dict(zip(dm_name, dm_vec))
        endtime = time.time()
        print("====End. Cost: {t:3f}====".format(t=endtime-starttime))
#        with open('D:/Python project/Plagiarism Project/DMVecModel.pl', 'wb') as dm_vec_model:
#            pickle.dump(dm_all_vec, dm_vec_model)
    
    test_root = "D:/dataset/_Plagiarism_docs_orig"
    test_dir_info = mytools.get_walkthrought_dir(test_root)
    test_list = []        
    test_dict = dict()
    for tt in test_dir_info:
        tdname = re.sub(test_root+r'\\', r'', str(tt[1]))
        tfname = re.sub(r'.txt', r'', str(tt[2]))
        ttname = tdname[:2]+'_'+tfname
        ttarget = ttname.split('_')[0]+'_'+ttname.split('_')[2]
        test_list.append(ttname)
        test_dict.update(dict({ttname:ttarget}))
    
    print("====Start test====")
    AR = 0
    count_AR = 0
    FDR = 0
    sorted_first_n = []
    sortthreshold = 500
    
    total_test_file = len(test_dir_info)
    starttime = time.time()
    for tdindex, td in enumerate(test_dir_info):
        with open(td[0], 'r', encoding='utf8') as ftest:
            test_content = ftest.read()
            test_vec, check = w2v_load.get_vector(test_content, model, weights_model)
        
        first_temp = dict()
        for dm_sour in dm_name:
            similarity_t = w2v_load.cosine_similarity(test_vec, dm_all_vec[str(dm_sour)])
            first_temp.update(dict({dm_sour:similarity_t}))
        sort_sim = sorted(first_temp.items(), key=lambda d: d[1], reverse=True)
        
        sorted_first_n = [i[0] for i in sort_sim[:sortthreshold]]
        target = str(test_dict[str(test_list[tdindex])])
        if target in sorted_first_n:
            count_AR += 1
            AR += sorted_first_n.index(target)
        else:
            FDR += 1
    endtime = time.time()
    
    print("AR:{ar:2f}".format(ar=AR/count_AR))
    print("FDR:{fdr:2f}%".format(fdr=FDR/total_test_file*100))
    print("CR:{cr:2f}".format(cr=AR/count_AR/(1-FDR/total_test_file)))
    print("====End. Each cost: {t:3f} sec====".format(t=(endtime-starttime)/total_test_file))
    print("====Total time: {t:3f} sec====".format(t=(endtime-starttime)))
    
    #del(dm_all_vec)
    pass