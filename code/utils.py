import jieba
import re
import ast
import numpy as np
import gensim


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


# if __name__ == '__main__':
# #     l1 = get_trainnums('../data/trainset.txt')
# #     print(l1)