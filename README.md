# DIAC_matching

## 比赛地址：https://biendata.com/competition/2019diac/ 

##方法记录：
###单折： 
随机初始化词向量+双向LSTM+Concat+softmax二分类 0.60 \
BERT maxlen=28(太小)  0.78 \
BERT maxlen=100     0.89 \
word2vec+ESIM       0.72 \
word2vec+ABCNN_3    0.68 \
word2vec+BIMPM      0.59 \ 
word2vec+DIIN


###5折cv：
word2vec+bilstm+concat （每折训练1 epoch） 0.5 \


### 改进
2.word2vec+BILSTM+Attention+concat+softmax \
3.word2vec+BILSTM+Attention+consine \
4.“Text Matching as Image Recognition” \
8.字符级别+词级别 \
9.Q和A分开
尝试方向：
ESIM最后得到p h向量后：
1取最后一层lstm拼接
2
