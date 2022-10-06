from gensim import corpora
from torch.autograd import Variable
import numpy as np
import torch
from data_util import config

# 要改!!9/12
def get_input_from_batch(batch, use_cuda):
  #初始化一些东西：enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage
  batch_size = len(batch.enc_lens)
  # print("get_batch_size", batch_size)
  enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
  # print("get_enc_batch",enc_batch.shape)
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
  enc_lens = batch.enc_lens
  extra_zeros = None
  enc_batch_extend_vocab = None


  lda_input=[]
  lda_input_o= batch.original_articles #<class 'list'>
  # print('-----------lda_input----------')
  # print('-----------lda_input----------',lda_input_o[0])
  # print('-----------lda_input----------',lda_input_o[1])
  # print('lda_input_o.len',len(lda_input_o))#2
  for i in range(len(lda_input_o)):
    lda_input_1 = [lda_input_o[i]]
    # print('type_lda_input_1',type(lda_input_1))
    lda_input.append(lda_input_1)
    # print('-----------lda_input----------',lda_input)
  lda_input = lda_input
  # print('-get——input——form——batch---lda_input-',lda_input)

  if config.pointer_gen:
    enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
      extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
  c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

  coverage = None
  if config.is_coverage:
    coverage = Variable(torch.zeros(enc_batch.size()))
    # print('coverage.shape',coverage.shape)
  if use_cuda:
    enc_batch = enc_batch.cuda()
    enc_padding_mask = enc_padding_mask.cuda()
    if enc_batch_extend_vocab is not None:
      enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
    if extra_zeros is not None:
      extra_zeros = extra_zeros.cuda()
    c_t_1 = c_t_1.cuda()
    if coverage is not None:
      coverage = coverage.cuda()
    # if config.is_lda is not None:
    #   lda_input = lda_input.cude()

  return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage,lda_input

def get_output_from_batch(batch, use_cuda):
  dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()
  target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

  if use_cuda:
    dec_batch = dec_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()
    dec_lens_var = dec_lens_var.cuda()
    target_batch = target_batch.cuda()

  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

