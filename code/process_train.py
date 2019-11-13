import utils

path_train = '../data/train_data.txt'
path_text = '../data/dev_set.csv'
path_stopword = '../data/哈工大停用词表.txt'
path_word2index = '../data/word2index.txt'
path_word = '../data/word.txt'
path_trainset = '../data/trainset.txt'


word2index = utils.get_word2index(path_word2index)

f1 = open(path_train, 'r', encoding='utf-8')
f2 = open(path_trainset, 'w', encoding='utf-8')
for x in f1.readlines():
    input1, input2, label = x.strip().split('\t')
    words1 = []
    words2 = []
    # sentence1
    for x in utils.tokenize(input1, path_stopword):
        if x in word2index:
            words1.append(word2index[x])
        else:
            words1.append(word2index['UNK'])
    # sentence2
    for x in utils.tokenize(input2, path_stopword):
        if x in word2index:
            words2.append(word2index[x])
        else:
            words2.append(word2index['UNK'])
    f2.write(str(words1)+'\t'+str(words2)+'\t'+label+'\n')
