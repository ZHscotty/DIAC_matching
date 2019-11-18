import gensim
import utils
import numpy as np

path_word2index = '../data/word2index.txt'
word2index = utils.get_word2index(path_word2index)
model = gensim.models.word2vec.Word2Vec.load('../result/model/w2v/word2vec_model')

index2word = {word2index[x]: x for x in word2index}
# test = [9, 12, 18, 20, 26]
# for x in test:
#     e = model.most_similar(positive=str(x), topn=10)
#     for xx in e:
#         print('test word:', index2word[str(x)], 'similar word:', index2word[xx[0]], 'rate:', xx[1])

emd = np.random.uniform(-0.05, 0.05, size=(len(word2index), 256))
matrix = model.wv.load_word2vec_format('../data/w2v')
for x in word2index:
    if word2index[x] in matrix:
        emd[int(word2index[x])] = matrix[word2index[x]]
print('emd_matrix shape:', emd.shape)