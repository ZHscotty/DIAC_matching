import gensim
import ast

# load train_data
train = []
with open('../data/w2v_train.txt', 'r', encoding='utf-8') as f:
    index = 0
    for x in f.readlines():
        train.append(ast.literal_eval(x))
model = gensim.models.word2vec.Word2Vec(sentences=train, min_count=3, size=256)
model.save('../result/model/w2v/word2vec_model')
model.wv.save_word2vec_format('../data/w2v', binary=False)