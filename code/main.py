from bilstm_classfication import Model
import config
import utils
import pandas as pd
import numpy as np

path_trainset = '../data/trainset.txt'
path_testset = '../data/testset.txt'
path_devset = '../data/devset.txt'
#
m = Model()
m.train(path_trainset, path_devset)
result = m.predict(path_testset)
id = utils.get_testid(path_testset)
output = [result, id]
output = np.array(output)
output = np.transpose(output)
print('output shape', output.shape)
d = pd.DataFrame(output)
d.to_csv('../result/submit/result.csv', encoding='utf-8', index=False, header=False)