import numpy as np
import jieba
import pandas as pd
from collections import Counter
import pickle
import re

path_train = '../data/train_data.txt'
path_text = '../data/dev_set.csv'
path_stopword = '../data/哈工大停用词表.txt'
path_word2index = '../data/word2index.txt'
path_word = '../data/word.txt'
word_list = []

# load stopword
with open(path_stopword, 'r', encoding='utf-8') as f:
    stopword = [x.strip() for x in f.readlines()]


# train_word
with open(path_train, 'r', encoding='utf-8') as f:
    for x in f.readlines():
        s1 = x.split('\t')[0]
        s2 = x.split('\t')[1]
        word_list.extend(jieba.lcut(s1))
        word_list.extend(jieba.lcut(s2))

# word_list = list(set(word_list))


# test_word
test = pd.read_csv(path_text, encoding='utf-8', delimiter='\t')
q1 = test['question1'].tolist()
q2 = test['question2'].tolist()
for x in q1:
    word_list.extend(jieba.lcut(x))

for x in q2:
    word_list.extend(jieba.lcut(x))

# remove stopword
word = []
for x in word_list:
    if x in stopword:
        pass
    elif len(re.findall('[0-9]+%*', x)) != 0:
        pass
    elif len(re.findall('\s+', x)) != 0:
        pass
    else:
        word.append(x)

print('len:', len(word))
word_list = word

# frequence of word
fre = dict(Counter(word_list))
fre = sorted(fre.items(), key=lambda x: x[1], reverse=True)
word_list = [x[0] for x in fre if x[1] > 2]

# create word2index
word2index = {x: index+1 for index, x in enumerate(word_list)}
word2index['UNK'] = 0

# save word list
with open(path_word, 'w', encoding='utf-8') as f:
    for x in word_list:
        f.write(str(x))
        f.write('\n')

# save word2index
with open(path_word2index, 'w', encoding='utf-8') as f:
    for x in word2index:
        f.write(str(x)+'\t'+str(word2index[x]))
        f.write('\n')







