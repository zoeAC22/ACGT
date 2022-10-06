import time

from gensim import corpora, models
import torch
# import nltk
from nltk import word_tokenize
from pytorch_transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from pytorch_transformers import BertModel
from nltk.corpus import stopwords
import string
from data_util import config


model_name = 'bert-base-uncased'
MODEL_PATH = 'D:\python\pointer\\bert-base-uncased\\'
# a. 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained('D:\python\pointer\\bert-base-uncased')
# b. 导入配置文件
model_config = BertConfig.from_pretrained('D:\python\pointer\\bert-base-uncased')
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
bert_model = BertModel.from_pretrained('D:\python\pointer\\bert-base-uncased\\')

model = models.LdaModel.load('lda.model')
dictionary=corpora.Dictionary.load('lda.dict')

# def LDA_big(train_txt):
train_txt =[["Sugar is bad to consume. My sister likes to have sugar, but not my father."],["Health experts say that Sugar is not good for your lifestyle."]]

final_output_0 = torch.zeros(1, 1, 768)
final_output_1 = torch.zeros(1, 1, 768)

def line_clean(line):
    # a = " ".join(line)
    lower = line.lower()
    # lower = line.lower()
    remove = str.maketrans('', '', string.punctuation)
    without_punctuation = lower.translate(remove)
    # print(without_punctuation)
    tokens = word_tokenize(without_punctuation)
    without_stopwords = [w for w in tokens if not w in stopwords.words('english')]
    # print(without_stopwords)
    return without_stopwords

for i in range(len(train_txt)):

    line_0 = train_txt[i]
    a = " ".join(line_0)
    line = line_clean(a)
    # print('line',line)
    # 得到 max
    bow = dictionary.doc2bow(line)
    # print(bow)
    t = model.get_document_topics(bow, minimum_probability=0.03)
    # print('t',t)
    aa=[]
    bb=[]
    for i in range(len(t)):
        aa.append(t[i][1])
        bb.append(t[i][0])
    # print(aa)
    max_0 = aa.index(max(aa))
    max_ = bb[max_0]
    # print(max_)
    str_1 = model.print_topic(max_, topn=3)

    list_2 = str_1.split('+')
    # print('list_2',list_2)
    LDA_words = []
    for w_str in list_2:
        # print('w_str',w_str)
        w_list = w_str.split('*')
        # print('w_list',w_list)
        new_0 = [w_list[1].replace('"', '').strip()]
        print('new_0',new_0)
        # LDA_words.append(w_list[1].replace('"', '').strip())
        # print('---------',LDA_words)
        f1_LDA_words_id = [tokenizer.convert_tokens_to_ids(new_0)]
        print(f1_LDA_words_id)
        tokens_tensor = torch.tensor(f1_LDA_words_id)
        print('tokens_tensor.shape',tokens_tensor.shape)
        output_bert_lda = bert_model(tokens_tensor)
        print(output_bert_lda[0].shape)
        final_output_0 = final_output_0 + output_bert_lda[0]
    final_output_1 = torch.cat((final_output_1, final_output_0), 0)
num = len(train_txt)
split_list = [1, num]
output = torch.split(final_output_1, split_list, dim=0)
z = output[1]
print(output[1].shape)
    # return z