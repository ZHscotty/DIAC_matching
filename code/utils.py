import jieba
import re
import ast
import numpy as np
import gensim
import pandas as pd
import os
from sklearn.model_selection import KFold

def get_word2index(path):
    word2index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for x in f.readlines():
            word, index = x.strip().split('\t')
            word2index[word] = index
    return word2index


def get_stopword(path):
    with open(path, 'r', encoding='utf-8') as f:
        stopword = [x.strip() for x in f.readlines()]
    return stopword


def tokenize(input, path_stopword):
    # input-str
    # path_stopword - stopword.txt
    stopword = get_stopword(path_stopword)
    words = []
    temp = jieba.lcut(input)
    for x in temp:
        if x in stopword:
            pass
        elif len(re.findall('[0-9]+%*', x)) != 0:
            pass
        elif len(re.findall('\s+', x)) != 0:
            pass
        else:
            words.append(x)
    return words


def get_maxlen(path):
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for x in f.readlines():
            s1, s2, _ = x.strip().split('\t')
            s1 = ast.literal_eval(s1)
            s2 = ast.literal_eval(s2)
            words.append(s1)
            words.append(s2)
    lent = [len(x) for x in words]

    return max(lent), sum(lent)/len(lent)

def get_testid(path):
    with open(path, 'r', encoding='utf-8') as f:
        ids = [x.strip().split('\t')[0] for x in f.readlines()]
    return ids


def get_trainnums(path):
    with open(path, 'r', encoding='utf-8') as f:
        return len(f.readlines())


def get_embeddingmatrix(path_word2index, path_w2vmodel, path_w2v):
    word2index = get_word2index(path_word2index)
    model = gensim.models.word2vec.Word2Vec.load(path_w2vmodel)
    emd = np.random.uniform(-0.05, 0.05, size=(len(word2index), 256))
    matrix = model.wv.load_word2vec_format(path_w2v)
    for x in word2index:
        if word2index[x] in matrix:
            emd[int(word2index[x])] = matrix[word2index[x]]
    print('emd_matrix shape:', emd.shape)
    return matrix


def process_tsv(path_tsv, path_testset):
    result = pd.read_csv(path_tsv, sep='\t', names=['0', '1']).values
    test_predict = np.argmax(result, axis=1)
    id = get_testid(path_testset)
    output = [id, test_predict]
    output = np.array(output)
    output = np.transpose(output)
    d = pd.DataFrame(output)
    d.to_csv('../result/submit/result.csv', encoding='utf-8', index=False, header=False, sep='\t')
    print('output ok!')


def get_maxchar(path_train, path_test):
    with open(path_train, 'r', encoding='utf-8') as f:
        lent = []
        for x in f.readlines():
            s1, s2, label = x.strip().split('\t')
            lent.append(len(s1))
            lent.append(len(s2))
    p = pd.read_csv(path_test, sep='\t')
    s1 = p['question1'].tolist()
    s2 = p['question2'].tolist()
    print(len(lent))
    for x in s1:
        lent.append(len(x))
        # print(x)
    for x in s2:
        lent.append(len(x))
    print(len(lent))
    return lent


def kfold(train_dir, k):
    file_path = train_dir+'/{}_fold'.format(k)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(os.path.join(train_dir, 'train_data.txt'), 'r', encoding='utf-8') as f:
        train_data = [line for line in f.readlines()]
    kfold = KFold(n_splits=k, shuffle=True, random_state=1)
    # print(os.path.join(os.path.join(train_dir, '{}_fold'.format(k)), 'train{}.txt'.format(0)))
    index = 0
    for train, test in kfold.split(train_data):
        f1 = open(os.path.join(os.path.join(train_dir, '{}_fold'.format(k)), 'train{}.txt'.format(index)), 'w',
                  encoding='utf-8')
        f2 = open(os.path.join(os.path.join(train_dir, '{}_fold'.format(k)), 'dev{}.txt'.format(index)), 'w',
                  encoding='utf-8')
        for x in train:
            f1.write(train_data[x])
        for x_ in test:
            f2.write(train_data[x_])
        f1.close()
        f2.close()
        index += 1


if __name__ == '__main__':
    # process_tsv(path_tsv='../result/submit/test_results.tsv', path_testset='../data/testset.txt')
    result_dir = '../result/submit/bert_result'
    tsvlist = os.listdir(result_dir)
    result = np.zeros(shape=(5000, 2))
    for x in tsvlist:
        path = os.path.join(result_dir, x)
        result += pd.read_csv(path, sep='\t', names=['0', '1']).values
    result = result/5
    test_predict = np.argmax(result, axis=1)
    id = get_testid('../data/testset.txt')
    output = [id, test_predict]
    output = np.array(output)
    output = np.transpose(output)
    d = pd.DataFrame(output)
    d.to_csv('../result/submit/result.csv', encoding='utf-8', index=False, header=False, sep='\t')
    print('output ok!')