from nltk.corpus import stopwords
import string
import gensim
import nltk
from gensim import corpora
filename='D:\python\pointer\\train.txt'#要处理的文档
fname='D:\pycharm\pointer_0\pointer\\training_ptr_gen'
file=open(filename,encoding='utf-8')
train=[]
i = 0
for line in file.readlines():
    lower = line.lower()
    remove = str.maketrans('', '', string.punctuation)
    without_punctuation = lower.translate(remove)
    tokens = nltk.word_tokenize(without_punctuation)
    without_stopwords = [w for w in tokens if not w in stopwords.words('english')]
    i = i + 1
    if i > 150000:#用150000句去做整个预料库
        break
    train.append(without_stopwords[0:len(without_stopwords)])

dictionary = corpora.Dictionary(train)
dictionary.save('lda.dict')
corpus = [dictionary.doc2bow(text) for text in train]
corpora.MmCorpus.serialize(fname+'corpus.mm',corpus)#保存corpus

Lda = gensim.models.ldamodel.LdaModel
lda = Lda(corpus, num_topics=43, id2word = dictionary,iterations = 10, chunksize = 10, passes = 5)
lda.save('lda.model')