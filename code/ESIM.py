import tensorflow as tf
import utils
import config
import ast
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.input1 = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input2 = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.label = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.MODEL_DIC = '../result/model/lstm_classfication'
        self.PIC_DIC = '../result/pic/lstm_classfication'
        self.word2index = utils.get_word2index('../data/word2index.txt')
        self.path_train = '../data/trainset.txt'
        self.path_test = '../data/testset.txt'
        self.trainnums = utils.get_trainnums(self.path_train)
        self.testnums = utils.get_trainnums(self.path_test)

        # embedding
        embedding_matrix = tf.get_variable(name='embedding', shape=[len(self.word2index), config.EMBEDD_SIZE])
        embedd_1 = tf.nn.embedding_lookup(embedding_matrix, self.input1)
        embedd_2 = tf.nn.embedding_lookup(embedding_matrix, self.input2)

        # batch,maxlen,size
        lstm_out1 = self.bilstm_layer(embedd_1, scope='lstm', reuse=False)
        lstm_out2 = self.bilstm_layer(embedd_2, scope='lstm', reuse=True)

        m1, m2 = self.attention_layer(lstm_out1, lstm_out2)
        print('m1 shape', m1.shape)

        lstm_1 = self.bilstm_layer(m1, scope='lstm_a', reuse=False)
        lstm_2 = self.bilstm_layer(m1, scope='lstm_b', reuse=False)
        print('lstm_1 shape', lstm_1.shape)

        v_a_avg = tf.reduce_mean(lstm_1, axis=1)
        v_b_avg = tf.reduce_mean(lstm_2, axis=1)
        v_a_max = tf.reduce_max(lstm_1, axis=1)
        v_b_max = tf.reduce_max(lstm_2, axis=1)

        v = tf.concat([v_a_avg, v_a_max, v_b_avg, v_b_max], axis=-1)
        print('v shape', v.shape)

        # dense
        self.logits = tf.layers.dense(v, units=2)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1)),
                                          dtype=tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))
        self.train_step = tf.train.AdamOptimizer(config.LR).minimize(self.loss)

    def bilstm_layer(self, inputs, scope=None, reuse=None):
        with tf.variable_scope('fw'+scope, reuse=reuse):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(config.LSTM_NUM)
        with tf.variable_scope('bw'+scope, reuse=reuse):
            cell_bw = tf.contrib.rnn.BasicLSTMCell(config.LSTM_NUM)
        with tf.name_scope('RNN_' + scope), tf.variable_scope('RNN_'+scope, reuse=reuse):
            lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs, dtype=tf.float32)
            lstm_out = tf.concat(lstm_out, 2)
        return lstm_out

    def attention_layer(self, input1, input2):
        # batch, maxlen, maxlen
        attentionWeights = tf.matmul(input1, tf.transpose(input2, [0, 2, 1]))
        attentionSoft_a = tf.nn.softmax(attentionWeights)
        attentionSoft_b = tf.nn.softmax(tf.transpose(attentionWeights))
        a_hat = tf.matmul(attentionSoft_a, input2)
        b_hat = tf.matmul(attentionSoft_b, input1)
        a_diff = tf.subtract(input1, a_hat)
        a_mul = tf.multiply(input1, a_hat)
        b_diff = tf.subtract(input2, b_hat)
        b_mul = tf.multiply(input2, b_hat)
        m_a = tf.concat([input1, a_hat, a_diff, a_mul], axis=-1)
        m_b = tf.concat([input2, b_hat, b_diff, b_mul], axis=-1)
        return m_a, m_b



    def train(self, train_path, dev_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            acc_train = []
            acc_dev = []
            loss_train = []
            loss_dev = []
            should_stop = False
            step = 0
            es_step = 0
            loss_stop = 99999
            n = 0

            while step < config.EPOCH and should_stop is False:
                print('Epoch:{}'.format(step))
                acc_total = 0
                loss_total = 0
                train_step = 0
                train_num = 0
                # trainset
                for input1, input2, label in self.prepare_data(train_path):
                    _, acc_t, loss_t = sess.run([self.train_step, self.acc, self.loss], {self.input1: input1,
                                                                                         self.input2: input2,
                                                                                         self.label: label,
                                                                                         self.is_training: True})
                    train_num += len(input1)
                    acc_total += acc_t
                    loss_total += loss_t
                    print('step{} [{}/{}]  --acc:{:.5f}, --loss:{:.5f}'.format(train_step, train_num,
                                                                               self.trainnums, acc_t, loss_t))
                    train_step += 1

                acc_t = acc_total / train_step
                loss_t = loss_total / train_step
                acc_train.append(acc_t)
                loss_train.append(loss_t)

                # devset
                acc_total = 0
                loss_total = 0
                dev_step = 0
                for input1, input2, label in self.prepare_data(dev_path):
                    acc_d, loss_d = sess.run([self.acc, self.loss], {self.input1: input1, self.input2: input2,
                                                                     self.label: label, self.is_training: True})
                    acc_total += acc_d
                    loss_total += loss_d
                    dev_step += 1
                acc_d = acc_total / dev_step
                loss_d = loss_total / dev_step
                acc_dev.append(acc_d)
                loss_dev.append(loss_d)
                print('Epoch{}----acc:{:.5f},loss:{:.5f},val_acc:{:.5f},val_loss:{:.5f}'.format(step, acc_t, loss_t,
                                                                                                acc_d, loss_d))

                loss_ = loss_d
                acc_ = acc_d

                # earlystop
                if loss_ > loss_stop:
                    if n >= config.EARLY_STEP:
                        should_stop = True
                    else:
                        n += 1
                else:
                    if not os.path.exists(self.MODEL_DIC):
                        os.makedirs(self.MODEL_DIC)
                    saver.save(sess, os.path.join(self.MODEL_DIC, 'model'))
                    es_loss = loss_
                    es_acc = acc_
                    es_step = step
                    n = 0
                    loss_stop = loss_
                step += 1
            if should_stop:
                print('Early Stop at Epoch{} acc:{} loss:{}'.format(es_step, es_acc, es_loss))

        if not os.path.exists(self.PIC_DIC):
            os.makedirs(self.PIC_DIC)

        plt.plot(acc_train)
        plt.plot(acc_dev)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.PIC_DIC, 'acc.png'))
        plt.close()

        plt.plot(loss_train)
        plt.plot(loss_dev)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.PIC_DIC, 'loss.png'))
        plt.close()

    def predict(self, test_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            result = []
            index = 0
            for input1, input2 in self.prepare_test(test_path):
                index += input1.shape[0]
                print('[{}/{}]'.format(index, self.testnums))
                logits = sess.run(self.logits, {self.input1: input1, self.input2: input2, self.is_training: False})
                result.append(logits)
            result = np.concatenate(result, axis=0)
            result = np.argmax(result, axis=1)
            print('result shape:', result.shape)
        return result

    def prepare_data(self, train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            examples = f.readlines()
            if len(examples)%config.BATCH_SIZE == 0:
                steps = len(examples)//config.BATCH_SIZE
            else:
                steps = len(examples) // config.BATCH_SIZE + 1
            begin = 0
            for i in range(steps):
                end = begin + config.BATCH_SIZE
                if end > len(examples):
                    end = len(examples)
                examples_batch = examples[begin:end]
                begin = end
                input1_batch = []
                input2_batch = []
                label_batch = []
                for x in examples_batch:
                    input1, input2, label = x.strip().split('\t')
                    input1 = ast.literal_eval(input1)
                    input2 = ast.literal_eval(input2)
                    label = to_categorical(label, 2)
                    input1_batch.append(input1)
                    input2_batch.append(input2)
                    label_batch.append(label)
                input1_batch = pad_sequences(input1_batch, maxlen=config.MAXLEN, padding='post')
                input2_batch = pad_sequences(input2_batch, maxlen=config.MAXLEN, padding='post')
                # input1_batch = np.array(input1_batch)
                # input2_batch = np.array(input2_batch)
                label_batch = np.array(label_batch)
                yield input1_batch, input2_batch, label_batch

    def prepare_test(self, test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            examples = f.readlines()
            if len(examples)%config.BATCH_SIZE == 0:
                steps = len(examples)//config.BATCH_SIZE
            else:
                steps = len(examples) // config.BATCH_SIZE + 1
            begin = 0
            for i in range(steps):
                end = begin + config.BATCH_SIZE
                if end > len(examples):
                    end = len(examples)
                examples_batch = examples[begin:end]
                begin = end
                input1_batch = []
                input2_batch = []
                for x in examples_batch:
                    qid, input1, input2 = x.strip().split('\t')
                    input1 = ast.literal_eval(input1)
                    input2 = ast.literal_eval(input2)
                    input1_batch.append(input1)
                    input2_batch.append(input2)
                input1_batch = pad_sequences(input1_batch, maxlen=config.MAXLEN, padding='post')
                input2_batch = pad_sequences(input2_batch, maxlen=config.MAXLEN, padding='post')
                yield input1_batch, input2_batch






