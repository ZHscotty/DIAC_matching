path_trainset = '../data/trainset.txt'
path_devset = '../data/devset.txt'
path_testset = '../data/testset.txt'
path_w2v = '../data/w2v_train.txt'

f1 = open(path_trainset, 'r', encoding='utf-8')
f2 = open(path_devset, 'r', encoding='utf-8')
f3 = open(path_testset, 'r', encoding='utf-8')
f4 = open(path_w2v, 'w', encoding='utf-8')

# trainset
for x in f1.readlines():
    q1, q2, _ = x.strip().split('\t')
    f4.write(q1+'\n'+q2+'\n')

# devset
for x in f2.readlines():
    q1, q2, _ = x.strip().split('\t')
    f4.write(q1 + '\n' + q2 + '\n')

# testset
for x in f3.readlines():
    _, q1, q2 = x.strip().split('\t')
    f4.write(q1 + '\n' + q2 + '\n')

f1.close()
f2.close()
f3.close()
f4.close()


# # trainset
# for x in data_list:
#     input1, input2, _ = x.strip().split('\t')
#     words1 = []
#     words2 = []
#     # sentence1
#     for x_ in utils.tokenize(input1, path_stopword):
#         if x_ in word2index:
#             words1.append(x_)
#         else:
#             words1.append(word2index['UNK'])
#     # sentence2
#     for x_ in utils.tokenize(input2, path_stopword):
#         if x_ in word2index:
#             words2.append(x_)
#         else:
#             words2.append(word2index['UNK'])
#     f2.write(str(words1)+'\n'+str(words2)+'\n')
#
# # test data
# p = pd.read_csv(path_text, encoding='utf-8', delimiter='\t')
# q1 = p['question1'].tolist()
# q2 = p['question2'].tolist()
# for x in q1:
#     words1 = []
#     for x_ in utils.tokenize(x, path_stopword):
#         if x_ in word2index:
#             words1.append(x_)
#         else:
#             words1.append(word2index['UNK'])
#     f2.write(str(words1)+'\n')
#
# for x in q2:
#     words2 = []
#     for x_ in utils.tokenize(x, path_stopword):
#         if x_ in word2index:
#             words2.append(x_)
#         else:
#             words2.append(word2index['UNK'])
#     f2.write(str(words2)+'\n')

