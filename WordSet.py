import string
import re

from textblob import TextBlob
from textblob import Word
from textblob.wordnet import Synset
from textblob.wordnet import VERB
from textblob.wordnet import NOUN
from statistics import mean

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer as wnl


def getWordSet(content):
    """Analysis article to get Verb_Set and Noun_Set for each sentence.  
    This version use TextBlob method to get wordset.  
    
    content : Input string of the article.  

    return [[VP_Set], [NP_Set]]: List of sentences which contain Verb_Set and Noun_Set.  

    """
    article = TextBlob(content)     # Load article into TextBlob.  
    article_sentence = []
    sent_count = 1

    # Use Textblob to separate sentences, and filter words by VERB and NOUN.  
    # Concatenate VP_sets and NN_sets in a list.  
    for sentence in article.sentences:
        sentUnit = TextBlob(str(sentence))
        sent_Tag = sentUnit.tags        # tags like "NN", "VP", "SUB", ..., etc, POS tag.  

        wordNN = []
        wordVP = []
        for tag in sent_Tag:
            if re.match('VP*', str(tag[1])):
                word = Word(str(tag[0]))
                wordVP.append(word.lemmatize())
            if re.match('NN*', str(tag[1])):
                word = Word(str(tag[0]))
                wordNN.append(word.lemmatize())
        wordVP = list(set(wordVP))      # Remove duplicate word.  
        wordNN = list(set(wordNN))
        sentPredicate = [wordVP, wordNN]

        # Check wordVP and wordNN whether the sets are null or not.  
        if wordVP and wordNN:
            print("{count}. {wordSet}".format(count=sent_count, wordSet=sentPredicate))
            sent_count += 1
            article_sentence.append(sentPredicate)
        else:
            pass
    return article_sentence


def getWordSet2(content):
    """Analysis article to get Verb_Set and Noun_Set for each sentence.  
    The version use NLTK method to get wordset.  
    
    content : Input string of the article.  

    return [[VP_Set], [NP_Set]]: List of sentences which contain Verb_Set and Noun_Set.  

    """
    article = nltk.sent_tokenize(content)     # Load article into nltk tokenizer to tokenize article to sentences.  
    article_sentence = []
    sent_count = 1

    # Use NLTK to separate sentences, and filter words by VERB and NOUN.  
    # Concatenate VP_sets and NN_sets in a list.  
    for sentence in article:
        sentWord = nltk.word_tokenize(sentence)
        sent_Tag = nltk.pos_tag(sentWord)        # tags like "NN", "VP", "SUB", ..., etc, POS tag.  

        wordNN = []
        wordVP = []
        for tag in sent_Tag:
            if re.match('VP*', str(tag[1])):
                word = Word(str(tag[0]))
                wordVP.append(word.lemmatize('v'))
            if re.match('NN*', str(tag[1])):
                word = Word(str(tag[0]))
                wordNN.append(word.lemmatize())

        wordVP = list(set(wordVP))
        wordNN = list(set(wordNN))
        sentPredicate = [wordVP, wordNN]

        # Check wordVP and wordNN whether the sets are null or not.  
        if wordVP and wordNN:
            print("{count}. {wordSet}".format(count=sent_count, wordSet=sentPredicate))
            sent_count += 1
            article_sentence.append(sentPredicate)
        else:
            pass
    return article_sentence


def wordSetSimilarity(wordSuspSet, wordSourSet):
    """Compare suspicious with sourse, and return the similarity of two wordset.  
    This version only sums up all similarity of each word in suspicious to source.  

    wordSuspSet : Wordset from suspicious.  
    wordSourSet : Wordset from source.     

    return : similarity score.  

    """

    total_similarity = 0
    threshold = 0.7

    for wordSusp in wordSuspSet:
        synsetSusp = wn.synsets(wordSusp)           # Get the instance of wordnet synset.
        for wordSour in wordSourSet:
            synsetSour = wn.synsets(wordFromSour)

            wordPairScore = []                      # The score for each synset compared pair.  
            for semanticWordSusp in synsetSusp:
                for semanticWordSour in synsetSoup:
                    if semanticWordSusp and semanticWordSour:
                        s = semanticWordSusp.wup_similarity(semanticWordSour)
                        if s is None or s < threshold:
                            pass
                        else:
                            wordList.append(s)
            if wordPairScore:
                maxPairScore = max(wordPairScore)
                #print('\"{0}\":\"{1}\" = {2}'.format(wordSusp, wordSour, maxPairScore))
                total_similarity += maxPairScore
            else:
                pass
    return total_similarity


def wordSetSimilarity2(wordSuspSet, wordSourSet, pos):
    """Compare suspicious with sourse, and return the similarity of two wordset.  
    This version divid into two POStag, VERB and NOUN, only compare the same POStag wordset.   

    wordSuspSet : Wordset from suspicious.  
    wordSourSet : Wordset from source.  
    pos : 'v' for VERB, 'n' for NOUN.  

    return : similarity score.  

    """
    total_similarity = 0
    threshold = 0.7

    if pos == 'v':
        POStag = VERB
    elif pos == 'n':
        POStag = NOUN
    else:
        return None

    for wordSusp in wordSuspSet:
        susp = Word(wordSusp)
        synsetSusp = susp.get_synsets(pos=POStag)      # Only get the specified POS synsets.  
        for wordSour in wordSourSet:
            sour = Word(wordSour)
            synsetSour = sour.get_synsets(pos=POStag)

            wordPairScore = []
            for semanticWordSusp in synsetSusp:
                for semanticWordSour in synsetSoup:
                    if semanticWordSusp and semanticWordSour:
                        s = semanticWordSusp.wup_similarity(semanticWordSour)
                        if s is None or s < threshold:
                            pass
                        else:
                            wordList.append(s)
            if wordPairScore:
                maxPairScore = max(wordPairScore)
                #print('\"{0}\":\"{1}\" = {2}'.format(wordSusp, wordSour, maxPairScore))
                total_similarity += maxPairScore
            else:
                pass
    return total_similarity


def wordSetSimilarity3(wordSuspSet, wordSourSet, pos):
    """Compare suspicious with sourse, and return the similarity of two wordset.  
    This version divid into two POStag, VERB and NOUN, only compare the same POStag wordset, 
    and filter the same word.

    wordSuspSet : Wordset from suspicious.  
    wordSourSet : Wordset from source.  
    pos : 'v' for VERB, 'n' for NOUN.  

    return : similarity score.  

    """
    total_similarity = 0
    threshold = 0.7

    if pos == 'v':
        POS = VERB
    elif pos == 'n':
        POS = NOUN
    else:
        return None

    setWordSusp = set(wordSuspSet)
    setWordSour = set(wordSourSet)

    sameWord = setWordSusp & setWordSour
    [setWordSusp.remove(element) for element in sameWord]
    [setWordSour.remove(element) for element in sameword]

    for wordSusp in wordSuspSet:
        susp = Word(wordSusp)
        synsetSusp = susp.get_synsets(pos=POS)

        susp2SourScore = []
        for wordSour in wordSourSet:
            sour = Word(wordSour)
            synsetSour = sour.get_synsets(pos=POS)

            wordPairScore = []
            for semanticWordSusp in synsetSusp:
                for semanticWordSour in synsetSoup:
                    if semanticWordSusp and semanticWordSour:
                        s = semanticWordSusp.wup_similarity(semanticWordSour)
                        if s is None or s < threshold:
                            pass
                        else:
                            wordList.append(s)
            if wordPairScore:
                maxPairScore = max(wordPairScore)
                #print('\"{0}\":\"{1}\" = {2}'.format(wordSusp, wordSour, maxPairScore))
                susp2SourScore.append(maxPairScore)
            else:
                pass

        if susp2SourScore:
            score = mean(susp2SourScore)
            total_similarity += score
            #print('\"{0}\" = {1]}'.format(wordSusp, wordSour))

    return total_similarity


if __name__ == '__main__':
    sent1 = 'I went to school.'
    sent2 = 'I run to school.'

    wordset1 = getWordSet2(sent1)
    wordset2 = getWordSet2(sent2)

    score = wordSetSimilarity2(wordset1[0], wordset2[0], pos='v')
    print(score)