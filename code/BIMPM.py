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
    def __init__(self, embedding_matrix=None):
        self.input1 = tf.placeholder(shape=[None, config.MAXLEN], dtype=tf.int32)
        self.input2 = tf.placeholder(shape=[None, config.MAXLEN], dtype=tf.int32)
        self.label = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.MODEL_DIC = '../result/model/bimpm'
        self.PIC_DIC = '../result/pic/bimpm'
        self.word2index = utils.get_word2index('../data/word2index.txt')
        self.path_train = '../data/trainset.txt'
        self.path_test = '../data/testset.txt'
        self.embedding_matrix = embedding_matrix
        self.trainnums = utils.get_trainnums(self.path_train)
        self.testnums = utils.get_trainnums(self.path_test)
        for i in range(1, 9):
            setattr(self, f'w{i}', tf.get_variable(name=f'w{i}', shape=(config.num_perspective, config.EMBEDD_SIZE),
                                                   dtype=tf.float32))

        """
        Word Representation Layer
        """
        # embedding
        if self.embedding_matrix is not None:
            embedding_matrix = tf.get_variable('embedding', [len(self.word2index), config.EMBEDD_SIZE],
                                               initializer=tf.constant_initializer(self.embedding_matrix))
        else:
            embedding_matrix = tf.get_variable('embedding', [1000, config.EMBEDD_SIZE],
                                               initializer=tf.random_normal_initializer)

        p_embedding = tf.nn.embedding_lookup(embedding_matrix, self.input1)
        h_embedding = tf.nn.embedding_lookup(embedding_matrix, self.input2)
        print('embedding:', p_embedding.shape)

        """ 
        Context Representation Layer
        """
        with tf.variable_scope("bilstm_p", reuse=tf.AUTO_REUSE):
            (p_fw, p_bw), _ = self.BiLSTM(p_embedding)
        with tf.variable_scope("bilstm_h", reuse=tf.AUTO_REUSE):
            (h_fw, h_bw), _ = self.BiLSTM(h_embedding)

        p_fw = self.dropout(p_fw)
        p_bw = self.dropout(p_bw)
        h_fw = self.dropout(h_fw)
        h_bw = self.dropout(h_bw)

        """
        Matching Layer
        """
        # 1.Full-Matching
        p_full_fw = self.full_matching(p_fw, tf.expand_dims(h_fw[:, -1, :], 1), self.w1)
        p_full_bw = self.full_matching(p_bw, tf.expand_dims(h_bw[:, 0, :], 1), self.w2)
        h_full_fw = self.full_matching(h_fw, tf.expand_dims(p_fw[:, -1, :], 1), self.w1)
        h_full_bw = self.full_matching(h_bw, tf.expand_dims(p_bw[:, 0, :], 1), self.w2)

        # 2、Maxpooling-Matching
        max_fw = self.maxpool_full_matching(p_fw, h_fw, self.w3)
        max_bw = self.maxpool_full_matching(p_bw, h_bw, self.w4)

        # 3、Attentive-Matching
        # 计算权重即相似度
        fw_cos = self.cosine(p_fw, h_fw)
        bw_cos = self.cosine(p_bw, h_bw)

        # 计算attentive vector
        p_att_fw = tf.matmul(fw_cos, h_fw)
        p_att_bw = tf.matmul(bw_cos, h_bw)
        h_att_fw = tf.matmul(fw_cos, p_fw)
        h_att_bw = tf.matmul(bw_cos, p_bw)

        p_mean_fw = tf.divide(p_att_fw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True))
        p_mean_bw = tf.divide(p_att_bw, tf.reduce_sum(bw_cos, axis=2, keep_dims=True))
        h_mean_fw = tf.divide(h_att_fw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True))
        h_mean_bw = tf.divide(h_att_bw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True))

        p_att_mean_fw = self.full_matching(p_fw, p_mean_fw, self.w5)
        p_att_mean_bw = self.full_matching(p_bw, p_mean_bw, self.w6)
        h_att_mean_fw = self.full_matching(h_fw, h_mean_fw, self.w5)
        h_att_mean_bw = self.full_matching(h_bw, h_mean_bw, self.w6)

        # 4、Max-Attentive-Matching
        p_max_fw = tf.reduce_max(p_att_fw, axis=2, keep_dims=True)
        p_max_bw = tf.reduce_max(p_att_bw, axis=2, keep_dims=True)
        h_max_fw = tf.reduce_max(h_att_fw, axis=2, keep_dims=True)
        h_max_bw = tf.reduce_max(h_att_bw, axis=2, keep_dims=True)

        p_att_max_fw = self.full_matching(p_fw, p_max_fw, self.w7)
        p_att_max_bw = self.full_matching(p_bw, p_max_bw, self.w8)
        h_att_max_fw = self.full_matching(h_fw, h_max_fw, self.w7)
        h_att_max_bw = self.full_matching(h_bw, h_max_bw, self.w8)

        mv_p = tf.concat(
            (p_full_fw, max_fw, p_att_mean_fw, p_att_max_fw,
             p_full_bw, max_bw, p_att_mean_bw, p_att_max_bw),
            axis=2)

        mv_h = tf.concat(
            (h_full_fw, max_fw, h_att_mean_fw, h_att_max_fw,
             h_full_bw, max_bw, h_att_mean_bw, h_att_max_bw),
            axis=2)


        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        mv_p = tf.reshape(mv_p, [-1, mv_p.shape[1], mv_p.shape[2] * mv_p.shape[3]])
        mv_h = tf.reshape(mv_h, [-1, mv_h.shape[1], mv_h.shape[2] * mv_h.shape[3]])

        """
        Aggregation Layer
        """
        with tf.variable_scope("bilstm_agg_p", reuse=tf.AUTO_REUSE):
            (p_f_last, p_b_last), _ = self.BiLSTM(mv_p)
        with tf.variable_scope("bilstm_agg_h", reuse=tf.AUTO_REUSE):
            (h_f_last, h_b_last), _ = self.BiLSTM(mv_h)

        x = tf.concat((p_f_last, p_b_last, h_f_last, h_b_last), axis=2)
        x = tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2]])

        x = self.dropout(x)

        x = tf.layers.dense(x, 10000, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 512)

        self.logits = tf.layers.dense(x, 2)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1)),
                                          dtype=tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))
        self.train_step = tf.train.AdamOptimizer(config.LR).minimize(self.loss)

    def BiLSTM(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(config.LSTM_NUM)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(config.LSTM_NUM)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def dropout(self, x):
        if self.is_training is True:
            return tf.nn.dropout(x, config.DROP)
        else:
            return x

    def full_matching(self, metric, vec, w):
        w = tf.expand_dims(tf.expand_dims(w, 0), 2)
        metric = w * tf.stack([metric] * config.num_perspective, axis=1)
        vec = w * tf.stack([vec] * config.num_perspective, axis=1)
        # 计算向量积
        m = tf.matmul(metric, tf.transpose(vec, [0, 1, 3, 2]))
        n = tf.norm(metric, axis=3, keep_dims=True) * tf.norm(vec, axis=3, keep_dims=True)
        cosine = tf.transpose(tf.divide(m, n), [0, 2, 3, 1])
        return cosine

    def maxpool_full_matching(self, v1, v2, w):
        cosine = self.full_matching(v1, v2, w)
        max_value = tf.reduce_max(cosine, axis=2, keep_dims=True)
        return max_value

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
            if len(examples) % config.BATCH_SIZE == 0:
                steps = len(examples) // config.BATCH_SIZE
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
            if len(examples) % config.BATCH_SIZE == 0:
                steps = len(examples) // config.BATCH_SIZE
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


# if __name__ == '__main__':
#     m = Model()




