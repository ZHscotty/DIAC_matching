# DIAC_matching

## [比赛地址](https://biendata.com/competition/2019diac/) 

##方法记录：
###单折： 
|实验方法|线上提交结果|model|
|---|---|---|
|word Embedding(random Initialization)+BILSTM|0.60|[bilstm_classfication](https://github.com/ZHscotty/DIAC_matching/blob/master/code/bilstm_classfication.py)|
|BERT maxlen=28(too small)|0.78|[bert](https://github.com/ZHscotty/DIAC_matching/blob/master/code/bert.py)|
|BERT maxlen=100|0.89|[bert](https://github.com/ZHscotty/DIAC_matching/blob/master/code/bilstm_classfication.py)|
|word Embeddding(word2vec)+ESIM|0.72|[ESIM](https://github.com/ZHscotty/DIAC_matching/blob/master/code/ESIM.py)|
|word Embeddding(word2vec)+ABCNN_3|0.68|[ABCNN](https://github.com/ZHscotty/DIAC_matching/blob/master/code/ABCNN.py)|
|word Embeddding(word2vec)+BIMPM|0.59|[BIMPM](https://github.com/ZHscotty/DIAC_matching/blob/master/code/BIMPM.py)|
|word Embeddding(word2vec)+DIIN|?|[DIIN](https://github.com/ZHscotty/DIAC_matching/blob/master/code/DIIN.py)|


###5折cv：
|实验方法|线上提交结果|
|---|---|
|word Embeddding(word2vec)+bilstm|0.5|
|BERT maxlen=100|0.9|


### 待实验
* word2vec+BILSTM+Attention
* charEmbedding+wordEmbedding
