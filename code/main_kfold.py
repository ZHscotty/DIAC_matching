from bilstm_classfication import Model
import config
import utils
import pandas as pd
import numpy as np
import os


path_testset = '../data/testset.txt'
path_word2index = '../data/word2index.txt'
path_w2vmodel = '../result/model/w2v/word2vec_model'
path_w2v = '../data/w2v'
model_dic = '../result/model'
pic_dic = '../result/pic'
#
embedding_matrix = utils.get_embeddingmatrix(path_word2index, path_w2vmodel, path_w2v)
k_fold = 5
result = np.zeros(shape=(5000, 2))
for i in range(k_fold):
    path_trainset = '../data/{}.fold/train{}.txt'.format(k_fold, i)
    path_devset = '../data/{}.fold/dev{}.txt'.format(k_fold, i)
    m = Model(model_dic=os.path.join(model_dic, 'bilstm_f{}'.format(i)),
              pic_dic=os.path.join(pic_dic, 'bilstm_f{}'.format(i)),
              embedding_matrix=embedding_matrix)
    m.train(path_trainset, path_devset)
    result += m.predict(path_testset)

result = result/5
result = np.argmax(result, axis=1)
print('result shape', result.shape)
id = utils.get_testid(path_testset)
output = [id, result]
output = np.array(output)
output = np.transpose(output)
print('output shape', output.shape)
d = pd.DataFrame(output)
d.to_csv('../result/submit/result.csv', encoding='utf-8', index=False, header=False, sep='\t')