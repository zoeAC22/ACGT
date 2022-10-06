#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
# import importlib
# importlib.reload(sys)

import os
import time

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher, data
from data_util.data import Vocab
from data_util import config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from train_util import get_input_from_batch

import rouge
from pprint import pprint

use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage
  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)
  @property
  def latest_token(self):
    return self.tokens[-1]
  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)

class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        print("decoder0000 model_name", model_name)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        self._rouge_article_dir = os.path.join(self._decode_dir, 'rouge_article_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir, self._rouge_article_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)

        print("decode_data_path00000000", config.decode_data_path)
        time.sleep(15)
        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        # print("0000000000")
        # print(batch)
        while batch is not None:

            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)
            # print("best_summary00000000000000000000000")
            # print(best_summary)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            # print("0000000000000kj")
            # print(output_ids)
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]
            original_article = batch.original_articles

            write_for_rouge(original_abstract_sents, decoded_words, original_article, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir, self._rouge_article_dir)
            counter += 1
            if counter % 10 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()
                break

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)

    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)
        # print("beam_search input")
        # print("enc_batch", enc_batch.shape)
        # print("enc_padding_mask", enc_padding_mask.shape)
        # print("enc_lens", enc_lens.shape)
        # print("enc_batch_extend_vocab", enc_batch_extend_vocab.shape)
        # print("extra_zeros", extra_zeros.shape)
        # print("c_t_0", c_t_0.shape)
        # print("coverage_t_0", coverage_t_0.shape)
        # print("c_t_LDA", c_t_LDA.shape)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        # print("beam_search encoder output")
        # print("encoder_outputs", encoder_outputs.shape)
        # print("encoder_feature", encoder_feature.shape)
        # print("encoder_hidden", len(encoder_hidden))

        # print("reduce state input")
        # print("encoder_hidden", encoder_hidden[0].shape)
        s_t_0 = self.model.reduce_state(encoder_hidden)
        # print("reduce_state output")
        # print("s_t_0", s_t_0[0].shape)
        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()
        # print("beam 99999999999")
        # print("dec_h[0]",dec_h[0].shape)
        # print("dec_c[0]", dec_c[0].shape)
        # print("c_t_0[0]", c_t_0[0].shape)
        # print("coverage_t_0[0]", coverage_t_0[0].shape)

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps_test and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            # print("y_t_1",y_t_1.shape)
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []
            all_context = []
            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)
            # print("all_state_h",all_state_h[0].shape)
            # print("all_state_c", all_state_c[0].shape)
            # print("all_context", all_context[0].shape)
            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)
            # print("s_t_1",s_t_1[0].shape)
            # print("c_t_1", c_t_1[0].shape)
            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)
            # print("all_coverage", all_coverage[0].shape)
            # print("coverage_t_1", coverage_t_1.shape)
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            # print("测试 解码 ")
            # print("decoder input")
            # print("y_t_1", y_t_1[0].shape)
            # print("s_t_1", s_t_1[0].shape)
            # print("encoder_outputs", encoder_outputs.shape)
            # print("encoder_feature", encoder_feature.shape)
            # print("enc_padding_mask", enc_padding_mask.shape)
            # print("extra_zeros", extra_zeros.shape)
            # print("enc_batch_extend_vocab", enc_batch_extend_vocab.shape)
            # print("c_t_1", c_t_1.shape)
            # print("coverage_t_1", coverage_t_1.shape)
            # print("steps", steps)
            # print("-----------------")
            # print("decoder output")
            # print("final_dist", final_dist.shape)
            # print("s_t", s_t[0].shape)
            # print("c_t", c_t[0].shape)
            # print("attn_dist", attn_dist.shape)
            # print("p_gen", p_gen.shape)
            # print("coverage_t", coverage_t.shape)
            log_probs = torch.log(final_dist)
            # print("log_probs", log_probs)
            # print(log_probs.shape)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)
            # print("topk_log_probs",topk_log_probs)
            # print(topk_log_probs.shape)
            # print("topk_ids",topk_ids)
            # print(topk_ids.shape)
            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()
            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)
                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)
            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break
            steps += 1
        if len(results) == 0:
            results = beams
        beams_sorted = self.sort_beams(results)
        return beams_sorted[0]

if __name__ == '__main__':
    # model_filename = sys.argv[1]
    beam_Search_processor = BeamSearch("../data/log_LDA/train_1607525963\model\model_5000_1608074152")
    beam_Search_processor.decode()


