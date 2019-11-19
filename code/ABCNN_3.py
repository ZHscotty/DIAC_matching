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
        self.MODEL_DIC = '../result/model/lstm_classfication'
        self.PIC_DIC = '../result/pic/lstm_classfication'
        self.word2index = utils.get_word2index('../data/word2index.txt')
        self.path_train = '../data/trainset.txt'
        self.path_test = '../data/testset.txt'
        self.embedding_matrix = embedding_matrix
        self.trainnums = utils.get_trainnums(self.path_train)
        self.testnums = utils.get_trainnums(self.path_test)

        # embedding
        if self.embedding_matrix is not None:
            embedding_matrix = tf.get_variable('embedding', [len(self.word2index), config.EMBEDD_SIZE],
                                               initializer=tf.constant_initializer(self.embedding_matrix))
        else:
            embedding_matrix = tf.get_variable('embedding', [1000, config.EMBEDD_SIZE],
                                               initializer=tf.random_normal_initializer)

        embedd_1 = tf.nn.embedding_lookup(embedding_matrix, self.input1)
        embedd_2 = tf.nn.embedding_lookup(embedding_matrix, self.input2)
        print('embedding:', embedd_1.shape)

        """ ABCNN_1 
        """
        p_embedding = tf.expand_dims(embedd_1, axis=-1)
        h_embedding = tf.expand_dims(embedd_2, axis=-1)

        # wide conv
        p_embedding = tf.pad(p_embedding, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])
        h_embedding = tf.pad(h_embedding, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])
        print('padding:', p_embedding.shape)

        # get euclidean distance
        euclidean = tf.sqrt(tf.reduce_sum(
            tf.square(tf.transpose(p_embedding, perm=[0, 2, 1, 3]) - tf.transpose(h_embedding, perm=[0, 2, 3, 1])),
            axis=1) + 1e-6)
        print('euclidean distance:', euclidean.shape)

        attention_matrix = 1 / (euclidean + 1)

        self.W0 = tf.get_variable(name="aW",
                                  shape=(config.MAXLEN + 4, config.EMBEDD_SIZE),
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004))

        p_attention = tf.expand_dims(tf.einsum("ijk,kl->ijl", attention_matrix, self.W0), -1)
        h_attention = tf.expand_dims(
            tf.einsum("ijk,kl->ijl", tf.transpose(attention_matrix, perm=[0, 2, 1]), self.W0), -1)
        print('attention:', p_attention.shape)

        p_embedding = tf.concat([p_embedding, p_attention], axis=-1)
        h_embedding = tf.concat([h_embedding, h_attention], axis=-1)
        print('p_embedding:', p_embedding.shape)

        p = tf.layers.conv2d(p_embedding, filters=64, kernel_size=(3, config.EMBEDD_SIZE))
        h = tf.layers.conv2d(h_embedding, filters=64, kernel_size=(3, config.EMBEDD_SIZE))
        print('p conv:', p.shape)

        p = tf.layers.average_pooling2d(p, (3, 1), strides=(1, 1))
        h = tf.layers.average_pooling2d(h, (3, 1), strides=(1, 1))
        print('p average(w-ap):', p.shape)

        if self.is_training is True:
            p = tf.nn.dropout(p, keep_prob=config.DROP)
            h = tf.nn.dropout(h, keep_prob=config.DROP)

        # p = tf.reshape(p, shape=[-1, config.MAXLEN, p.shape[2]*p.shape[3]])
        # h = tf.reshape(h, shape=[-1, config.MAXLEN, h.shape[2]*h.shape[3]])
        # print('p_1:', p.shape)
        #
        # p = tf.expand_dims(p, axis=3)
        # h = tf.expand_dims(h, axis=3)

        """ABCNN_2"""
        attention_pool_euclidean = tf.sqrt(
            tf.reduce_sum(tf.square(tf.transpose(p, perm=[0, 3, 1, 2]) - tf.transpose(h, perm=[0, 3, 2, 1])),
                          axis=1))
        print('attention_pool_euclidean:', attention_pool_euclidean.shape)

        attention_pool_matrix = 1 / (attention_pool_euclidean + 1)
        p_sum = tf.reduce_sum(attention_pool_matrix, axis=2, keep_dims=True)
        h_sum = tf.reduce_sum(attention_pool_matrix, axis=1, keep_dims=True)
        print('p_sum', p_sum.shape)
        print('h_sum', h_sum.shape)

        p = tf.reshape(p, shape=(-1, p.shape[1], p.shape[2] * p.shape[3]))
        h = tf.reshape(h, shape=(-1, h.shape[1], h.shape[2] * h.shape[3]))
        print('p new', p.shape)

        p = tf.multiply(p, p_sum)
        h = tf.multiply(h, tf.matrix_transpose(h_sum))
        print('p', p.shape)

        p = tf.expand_dims(p, axis=3)
        h = tf.expand_dims(h, axis=3)
        print('p expand', p.shape)

        p = tf.pad(p, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])
        h = tf.pad(h, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])
        print('p padding', p.shape)

        p = tf.layers.conv2d(p, filters=64, kernel_size=(3, 64))
        h = tf.layers.conv2d(h, filters=64, kernel_size=(3, 64))
        print('p conv2', p.shape)

        if self.is_training is True:
            p = tf.nn.dropout(p, keep_prob=config.DROP)
            h = tf.nn.dropout(h, keep_prob=config.DROP)

        # average pooling(all-ap)
        p = tf.layers.average_pooling2d(p, pool_size=(p.shape[1], 1), strides=(1, 1))
        h = tf.layers.average_pooling2d(h, pool_size=(h.shape[1], 1), strides=(1, 1))
        print('p averagepool:', p.shape)

        x = tf.concat((p, h), axis=-1)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]*x.shape[3]))
        print('x:', x.shape)

        out = tf.layers.dense(x, 50)
        self.logits = tf.layers.dense(out, 2)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1)),
                                          dtype=tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))
        self.train_step = tf.train.AdamOptimizer(config.LR).minimize(self.loss)

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




