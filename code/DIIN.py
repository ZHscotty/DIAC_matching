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
        self.gate_w1 = tf.get_variable(name='gate_w1', shape=(config.EMBEDD_SIZE * 2, config.EMBEDD_SIZE))
        self.gate_w2 = tf.get_variable(name='gate_w2', shape=(config.EMBEDD_SIZE * 2, config.EMBEDD_SIZE))
        self.gate_w3 = tf.get_variable(name='gate_w3', shape=(config.EMBEDD_SIZE * 2, config.EMBEDD_SIZE))
        self.gate_b1 = tf.get_variable(name='gate_b1', shape=(config.EMBEDD_SIZE,))
        self.gate_b2 = tf.get_variable(name='gate_b2', shape=(config.EMBEDD_SIZE,))
        self.gate_b3 = tf.get_variable(name='gate_b3', shape=(config.EMBEDD_SIZE,))

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
        p_encode = self.encode(p_embedding, 'p_encode')
        h_encode = self.encode(h_embedding, 'h_encode')

        p_encode = self.dropout(p_encode)
        h_encode = self.dropout(h_encode)

        I = tf.multiply(tf.expand_dims(p_encode, axis=2), tf.expand_dims(h_encode, axis=1))
        print('I', I.shape)

        dense_out = self.dense_net(I)
        dense_out = self.dropout(dense_out)

        dense_out = tf.reshape(dense_out, shape=(-1, dense_out.shape[1] * dense_out.shape[2] * dense_out.shape[3]))
        out = tf.layers.dense(dense_out, 256)
        out = self.dropout(out)

        self.logits = tf.layers.dense(out, 2)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1)),
                                          dtype=tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))
        self.train_step = tf.train.AdamOptimizer(config.LR).minimize(self.loss)

    def dropout(self, x):
        if self.is_training is True:
            return tf.nn.dropout(x, config.DROP)
        else:
            return x

    def encode(self, v, scope):
        v_concat = tf.concat((v, v, tf.multiply(v, v)), axis=-1)
        v_att = self.attention_layer(scope_name=scope, input_tensor=v_concat, input_size=v_concat.shape[-1],
                                     output_size=config.attention_size)
        p_concat = tf.concat((v, v_att), axis=-1)
        z = tf.nn.tanh(tf.einsum("ijk,kl->ijl", p_concat, self.gate_w1) + self.gate_b1)
        r = tf.nn.sigmoid(tf.einsum("ijk,kl->ijl", p_concat, self.gate_w2) + self.gate_b2)
        f = tf.nn.sigmoid(tf.einsum("ijk,kl->ijl", p_concat, self.gate_w3) + self.gate_b3)
        res = tf.multiply(r, v) + tf.multiply(f, z)
        return res

    def attention_layer(self, scope_name, input_tensor, input_size, output_size):
        with tf.variable_scope(scope_name):
            # Q
            wq = tf.get_variable('weightq', [input_size, output_size])
            bq = tf.get_variable('biasq', [output_size], initializer=tf.zeros_initializer)
            q = tf.einsum('aij,jk->aik', input_tensor, wq) + bq

            # K
            wk = tf.get_variable('weightk', [input_size, output_size])
            bk = tf.get_variable('biask', [output_size], initializer=tf.zeros_initializer)
            k = tf.einsum('aij,jk->aik', input_tensor, wk) + bk

            # V
            wv = tf.get_variable('weightv', [input_size, output_size])
            bv = tf.get_variable('biasv', [output_size], initializer=tf.zeros_initializer)
            v = tf.einsum('aij,jk->aik', input_tensor, wv) + bv

            # Q*K/
            k = tf.transpose(k, perm=[0, 2, 1])
            # shape(batch, maxlen, maxlen)
            input_score = tf.einsum('aij,ajk->aik', q, k)
            d = tf.sqrt(float(output_size))
            input_score = input_score / d
            input_score = tf.nn.softmax(input_score, axis=2)

            # get weighted value
            # shape(batch, maxlen, 2*lstm_hidden_size)
            att_output = tf.einsum('aij,ajk->aik', input_score, v)

            return att_output

    def dense_net(self, v):
        filters = v.shape[-1] * 0.5
        v_in = tf.layers.conv2d(v, filters=filters, kernel_size=(1, 1))
        for _ in range(3):
            for _ in range(8):
                v_out = tf.layers.conv2d(v_in,
                                         filters=20,
                                         kernel_size=(3, 3),
                                         padding='SAME',
                                         activation='relu')
                v_in = tf.concat((v_in, v_out), axis=-1)
            transition = tf.layers.conv2d(v_in,
                                          filters=int(v_in.shape[-1].value * 0.5),
                                          kernel_size=(1, 1))
            transition_out = tf.layers.max_pooling2d(transition,
                                                     pool_size=(2, 2),
                                                     strides=2)
            v_in = transition_out
        return v_in

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




