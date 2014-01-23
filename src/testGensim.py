#-*- coding:UTF-8 -*-
'''
Created on Dec 24, 2013

@author: HAO
'''
import jieba
from gensim import corpora, models, similarities

# import pymongo
# conn = pymongo.Connection('localhost', 27017);
# db = conn.test
# dbCollection = db.classifier

sentences = []
sentences.append("【我国乙肝疫苗接种13年 接种后死亡188例】昨天下午，国家食药监总局、卫生计生委联合召开发布会，就社会关注的乙肝疫苗疑似致死事件进行通报。根据中国疾控中心数据，从2000年到今年12月，接种乙肝疫苗后死亡的疑似异常反应病例已上报188例，最终确定为疫苗异常反应的18例。")
sentences.append("【呵护微笑天使】江豚，目前我国淡水水域中唯一的胎生哺乳动物，由于栖息地的丧失，生存水质被污染和破坏，2006年到2012年，长江江豚平均每72小时就消失一头，任此趋势发展下去，长江江豚最快将在10年内灭绝。保护江豚，别让江豚微笑告别。[心]大家一起转发努力。")
sentences.append("【步Facebook和Twitter后尘，伊朗政府屏蔽微信】伊朗政府负责监管网络内容的部门发言称，将在全国范围内屏蔽微信。Global Voices通过对伊朗当地网络用户的确认，证实微信在当地已经无法使用。当然还是可以翻墙。Facebook和Twitter此前同样遭到屏蔽。")
sentences.append("【台统派:大陆是台湾希望 强盛做得好统一会愈快】台湾中国统一联盟桃竹苗荣誉分会长阎中杰说，中国一定强，台湾才有希望，美国对台湾的友善是假的，利用我们买飞机大炮。大陆强盛做得好，台湾更认同，更有利两岸统一，可以领导台湾，大陆愈改愈好，相信两岸统一会愈快。")
sentences.append("HR吐血整理的面试技巧宝典 一起学习下，争取明年找个好工作！")

words=[]
for doc in sentences:
    words.append(list(jieba.cut(doc)))
# print words

dic = corpora.Dictionary(words)
# print dic
# print dic.token2id
# for word,index in dic.token2id.iteritems():
#     print word +" 编号为:"+ str(index)
 
# Corpus is simply an object which, when iterated over, returns its documents represented as sparse vectors.
# （9，2）这个元素代表第二篇文档中id为9的单词出现了2次。
corpus = [dic.doc2bow(text) for text in words]
# print corpus

tfidf = models.TfidfModel(corpus)
# vec = [(0, 1), (4, 1)]
# print tfidf[vec]
corpus_tfidf = tfidf[corpus]
# for doc in corpus_tfidf:
#     print doc

# index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=10)
# sims = index[tfidf[vec]]
# print list(enumerate(sims))

# 训练lsi模型
lsi = models.LsiModel(corpus_tfidf, id2word=dic, num_topics=3)
lsiout=lsi.print_topics(2)
# print lsiout[0]
# print lsiout[1]
# dbCollection.insert({"classifier":"lsiModel","content":lsiout})
corpus_lsi = lsi[corpus_tfidf]
# for doc in corpus_lsi:
#     print doc

# # 训练lda模型
# lda = models.LdaModel(corpus_tfidf, id2word=dic, num_topics=3)
# ldaOut=lda.print_topics(2)
# print ldaOut[0]
# print ldaOut[1]
# corpus_lda = lda[corpus_tfidf]
# for doc in corpus_lda:
#     print doc
# 搜索
index = similarities.MatrixSimilarity(lsi[corpus])
query = sentences[0]
query_bow = dic.doc2bow(list(jieba.cut(query)))
print query_bow
print tfidf[query_bow]

query_lsi = lsi[query_bow]
print query_lsi

# 计算其和index中doc的余弦相似度了：

sims = index[query_lsi]
print list(enumerate(sims))
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print sort_sims
