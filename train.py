from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import tensorflow as tf
import torch
from training_ptr_gen.model import Model
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad
from data_util import config
from data_util.batcher import Batcher, Example
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss

from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch
from training_ptr_gen.decode import BeamSearch
use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    print('开始Train：')

    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)
        print("train_data_path", config.train_data_path)
        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        print("存放模型的文件夹", train_dir)

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        print("self.model_dir（路径连接模型）", self.model_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.compat.v1.summary.FileWriter(train_dir)  # 将训练日志（过程，数据）写在train_dir里面
        # self.summary_writer = tf.summary.create_file_writer(train_dir)
        # self.summary_writer = tf.summary.reexport_tf_summary(train_dir)
        self.R = ""

    print('开始save_model：')
    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        # reduce_state是
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        self.R = model_save_path
        print("save_model_path：", model_save_path)
        torch.save(state, model_save_path)

    print('开始setup_train：')
    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())

        total_params = sum([param[0].nelement() for param in params])
        print('The Number of params of model: %.3f million' % (total_params / 1e6))  # million
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']
            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    print('开始train_one_batch：')
    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, lda_input = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        # 从encoder端到decoder端的输入，并且初始化了一个decoder端的输出。
        # print("reduce_state output")
        # print("s_t_1",s_t_1[0].shape)
        # print('开始generated ：')
        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing 初始化的
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask, c_t_1,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage, di, lda_input)


            target = target_batch[:, di]

            # print("target")
            # print(target.shape)
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            # print("gold_probs",gold_probs)
            # print(gold_probs.shape)

            #计算损失：
            step_loss = -torch.log(gold_probs + config.eps)

            # step_loss =

            # print("step_loss",step_loss)
            # print(step_loss.shape)

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                # print("coverage loss")
                # print("step_coverage_loss",step_coverage_loss)
                # print(step_coverage_loss.shape)

                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                # print("step_loss", step_loss)
                # print(step_loss.shape)

                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            # print("step_mask",step_mask)
            # print(step_mask.shape)
            # print("step_loss           ",step_loss)
            # print(step_loss.shape)
            step_loss = step_loss * step_mask
            # print("step_loss@@@@", step_loss)
            # print(step_loss.shape)
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # print("sum_losses",sum_losses)
        # print(sum_losses.shape)
        batch_avg_loss = sum_losses / dec_lens_var
        # print("batch_avg_loss",batch_avg_loss)
        # print(batch_avg_loss.shape)
        loss = torch.mean(batch_avg_loss)
        # print("loss", loss)
        # print(loss.shape)

        loss.backward()

        # 梯度裁剪：
        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    print('开始trainIters ：')
    def trainIters(self, n_iters, model_file_path=None):
        # setup_train(model_file_path)：return start_iter, start_loss
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            print('while iter < n_iters ---> batch = self.batcher.next_batch')
            batch = self.batcher.next_batch()
            # print(type(batch))
            loss = self.train_one_batch(batch)
            # print('trainIters loss:')
            # print(len(loss))
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1
            if iter % 100 == 0:
                # 画图结果
                self.summary_writer.flush()
            print_interval = 1
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                with open("train_loss_test.txt", "a", encoding="utf-8") as f_w:
                    f_w.writelines('steps %d, seconds for %d batch: %.2f , loss: %f\n' % (iter, print_interval,
                                                                                          time.time() - start, loss))
                start = time.time()
            # 每一个保存一次模型！
            if iter % 1 == 0:  # 1000
                self.save_model(running_avg_loss, iter)
                print("--------------------------------")
                # print("Begin to generate:")
                # beam_Search_processor = BeamSearch(self.R)
                # beam_Search_processor.decode()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        # default= None,
                        default='',
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
