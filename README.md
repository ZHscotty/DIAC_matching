# DIAC_matching
比赛地址：https://biendata.com/competition/2019diac/ \
方法记录：\
单折：
随机初始化词向量+双向LSTM+Concat+softmax二分类 0.60 \
BERT maxlen=28(太小) 0.78 \
word2vec+ESIM 0.72 \
word2vec+ABCNN_3  \
BERT maxlen=100  \


5折：




2.word2vec+BILSTM+Attention+concat+softmax \
3.word2vec+BILSTM+Attention+consine \
4.“Text Matching as Image Recognition” \
8.字符级别+词级别 \
9.Q和A分开
