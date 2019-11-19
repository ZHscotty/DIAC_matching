import utils
import pandas as pd

path_train = '../data/train_data.txt'
path_text = '../data/dev_set.csv'
path_stopword = '../data/哈工大停用词表.txt'
path_word2index = '../data/word2index.txt'
path_word = '../data/word.txt'
path_trainset = '../data/trainset.txt'
path_testset = '../data/testset.txt'


word2index = utils.get_word2index(path_word2index)
test = pd.read_csv(path_text, sep='\t')
qid = test['qid'].tolist()
s1 = test['question1'].tolist()
s2 = test['question2'].tolist()
f = open(path_testset, 'w', encoding='utf-8')
for x in range(len(s1)):
    words1 = []
    words2 = []
    # sentence1
    for y in utils.tokenize(s1[x], path_stopword):
        if y in word2index:
            words1.append(word2index[y])
        else:
            words1.append(word2index['UNK'])

    # sentence1
    for y in utils.tokenize(s2[x], path_stopword):
        if y in word2index:
            words2.append(word2index[y])
        else:
            words2.append(word2index['UNK'])
    f.write(str(qid[x])+'\t'+str(words1)+'\t'+str(words2)+'\n')
f.close()
