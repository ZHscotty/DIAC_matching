import jieba
import re
import ast
import os
import numpy as np
import gensim
from sklearn.model_selection import KFold


# 获取Word2index
def get_word2index(path):
    word2index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for x in f.readlines():
            word, index = x.strip().split('\t')
            word2index[word] = index
    return word2index


# 加载停用词表
def get_stopword(path):
    with open(path, 'r', encoding='utf-8') as f:
        stopword = [x.strip() for x in f.readlines()]
    return stopword


# 分词
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


# 获取words级别的最大长度
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


# 获取测试集的id信息
def get_testid(path):
    with open(path, 'r', encoding='utf-8') as f:
        ids = [x.strip().split('\t')[0] for x in f.readlines()]
    return ids


# 获取训练集的总数
def get_trainnums(path):
    with open(path, 'r', encoding='utf-8') as f:
        return len(f.readlines())


# 获取预训练词向量
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


# k-fold
def kfold(train_dir, k):
    os.makedirs(path=os.path.join(train_dir, '/{}_fold'.format(k)))
    with open(os.path.join(train_dir, 'train_data.txt'), 'r', encoding='utf-8') as f:
        train_data = [line for line in f.readlines()]
    kfold = KFold(n_splits=k, shuffle=True, random_state=1)

    for index in range(k):
        f1 = open(os.path.join(os.path.join(train_dir, '/{}_fold'.format(k)), '/train{}'.format(index)))
        f2 = open(os.path.join(os.path.join(train_dir, '/{}_fold'.format(k)), '/dev{}'.format(index)))
        for train, test in kfold.split(train_data):
            for x in train_data[train]:
                f1.write(x)
            for x_ in train_data[test]:
                f2.write(x_)
        f1.close()
        f2.close()



# if __name__ == '__main__':
# #     l1 = get_trainnums('../data/trainset.txt')
# #     print(l1)