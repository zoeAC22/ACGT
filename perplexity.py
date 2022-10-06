#-*-coding:utf-8-*-
import math
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import os
from gensim.corpora import Dictionary
from gensim import corpora, models
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s : ', level=logging.INFO)
fname='D:\pycharm\pointer_0\pointer\\training_ptr_gen'

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    """calculate the perplexity of a lda-model"""
    print ('the info of this ldamodel: \n')
    print ('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0 # the num of words in the doc
        doc=dict(doc)
        for word_id, num in doc.items():
            prob_word = 0.0 # the probablity of the word
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    print ("the perplexity of this ldamodel is : %s"%prep)
    return prep
if __name__ == '__main__':
    model = models.LdaModel.load('lda.model')
    dictionary = corpora.Dictionary.load('lda.dict')
    corpus_path = fname + 'corpus.mm'
    corpus=corpora.MmCorpus(corpus_path)
    lda_multi = models.ldamodel.LdaModel.load('lda.model')
    num_topics = 50
    testset = []
    # sample 1/300
    for i in range(int(corpus.num_docs/300)):
        testset.append(corpus[i*300])
    prep = perplexity(lda_multi, testset, dictionary, len(dictionary.keys()), num_topics)
    print(prep)
