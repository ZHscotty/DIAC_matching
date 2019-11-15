import utils
from sklearn.model_selection import train_test_split

path_train = '../data/train_data.txt'
path_text = '../data/dev_set.csv'
path_stopword = '../data/哈工大停用词表.txt'
path_word2index = '../data/word2index.txt'
path_word = '../data/word.txt'
path_trainset = '../data/trainset.txt'
path_devset = '../data/devset.txt'


word2index = utils.get_word2index(path_word2index)

f1 = open(path_train, 'r', encoding='utf-8')
f2 = open(path_trainset, 'w', encoding='utf-8')
f3 = open(path_devset, 'w', encoding='utf-8')
data_list = f1.readlines()

data_train, data_dev = train_test_split(data_list, random_state=40, test_size=0.3)

for x in data_train:
    input1, input2, label = x.strip().split('\t')
    words1 = []
    words2 = []
    # sentence1
    for x_ in utils.tokenize(input1, path_stopword):
        if x_ in word2index:
            words1.append(word2index[x_])
        else:
            words1.append(word2index['UNK'])
    # sentence2
    for x_ in utils.tokenize(input2, path_stopword):
        if x_ in word2index:
            words2.append(word2index[x_])
        else:
            words2.append(word2index['UNK'])
    f2.write(str(words1)+'\t'+str(words2)+'\t'+label+'\n')

for x in data_dev:
    input1, input2, label = x.strip().split('\t')
    words1 = []
    words2 = []
    # sentence1
    for x_ in utils.tokenize(input1, path_stopword):
        if x_ in word2index:
            words1.append(word2index[x_])
        else:
            words1.append(word2index['UNK'])
    # sentence2
    for x_ in utils.tokenize(input2, path_stopword):
        if x_ in word2index:
            words2.append(word2index[x_])
        else:
            words2.append(word2index['UNK'])
    f3.write(str(words1)+'\t'+str(words2)+'\t'+label+'\n')

f1.close()
f2.close()
f3.close()