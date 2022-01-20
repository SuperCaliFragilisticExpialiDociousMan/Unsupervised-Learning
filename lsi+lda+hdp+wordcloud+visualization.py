#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:42:29 2021

@author: shizhengyan
"""
import spacy 
from spacy import displacy
import os
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel, LsiModel, HdpModel
#from gensim.models.wrappers import LdaMallet


test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
print(test_data_dir)
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
print(lee_train_file)
text = open(lee_train_file).read()

print(len(text))

##############################
'clean data'
# We can't have state-of-the-art results without data which is as good. 
#Let's spend this section working on cleaning and understanding our data set. 
#We will be checking out spacy, an industry grade text-processing package.

nlp=spacy.load('en_core_web_sm')



#For safe measure, let's add some stopwords. 
#It's a newspaper corpus, so it is likely we will be coming across variations of 'said', 'Mister', and 'Mr'... 
#which will not really add any value to the topic models.

my_stop_words = ['say', '\s', 'mr', 'Mr', 'said', 'says', 'saying', 'today', 'be']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

doc=nlp(text)

words=[]
for i in range(len(doc)):
    words.append(str(doc[i]))

#print(len(doc))

print('words: ',words)

#print('words: ',words)

f=0
for i in words:
    if i=='of':
        f+=1

print('f: ',f)

from collections import Counter

c=Counter(words)
c=list(c.items())
c=sorted(c,key=lambda x: x[1], reverse=True)
#print(c)


from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms 

'''
def clean(words):
 
    stemmer=PorterStemmer()
    tokenizer = TweetTokenizer()

    cleaned = []
    Stemmer=dict()
    for word in words:
        stem=stemmer.stem(word)
        if word not in stopwords.words('english') and stem not in Stemmer:
            #word=stemmer.stem(word)
            cleaned.append(word)
            Stemmer[stem]=word
        elif word not in stopwords.words('english') and stem in Stemmer:
            cleaned.append(Stemmer[stem])
    return cleaned
'''

'''
W=Counter(words)
W=list(W.items())
W=sorted(W,key=lambda x: x[1], reverse=True)
print('frenquency of cleaned words: ',W)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12.5,8))
x=['say','australia','year','new','palestinian','government','south','attack','day','state']
y=[1013,338,182,177,173,151,142,142,138,129]
sns.set(style="darkgrid")
# 构建数据
tips = sns.load_dataset("tips")

sns.barplot(x=x, y=y)
plt.show()


plt.figure(figsize=(12.5,8))
x=['the','.',',','to','of','in','and','a','The','is']
y=[3532,2682,2396,1682,1534,1306,1237,1169,603,572]
color=['purple','green']
plt.pie(y,labels=x,autopct='%1.1f%%',shadow=False,counterclock = False, wedgeprops = {'width' : 0.4})
plt.grid(True,linestyle=':',color='r',alpha=0.6)
plt.show()


plt.figure(figsize=(12.5,8))
x=['say','australia','year','new','palestinian','government','south','attack','day','state']
y=[1013,338,182,177,173,151,142,142,138,129]
color=['red','black','purple','green','blue','peru','orchid','deepskyblue','orange','pink']
plt.bar(x,y,color=color)
plt.grid(True,linestyle=':',color='r',alpha=0.6)
plt.show()
'''



sent = nlp('Last Thursday, Manchester United defeated AC Milan at San Siro.')

for token in sent:
    print(token.text, token.pos_, token.tag_)
    

texts, article = [], []

for word in doc:
    
    if word.text != '\n' and not word.is_stop and not word.is_punct and not word.like_num and word.text != 'I':
        article.append(word.lemma_)
        
    if word.text == '\n':
        texts.append(article)
        article = []

print(texts[0])

bigram = gensim.models.phrases.Phrases(texts)
texts = [bigram[line] for line in texts]


from gensim import corpora, models

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]



###############################################################

'LSI'

print('This part is LSI')

lsi_model = LsiModel(corpus=corpus, num_topics=5, id2word=dictionary)

lsi_result=lsi_model.show_topics(num_topics=5)
for i in lsi_result:
    print(i)


#################################################################3




'LDA'

print('Last part is LDA')
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)
#lda_model = LdaModel(corpus=corpus, num_topics=5)
print(lda_model.show_topics())


#import pyLDAvis.gensim_models

#lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus_tfidf, dictionary, sort_topics = False)
#print(pyLDAvis.display(lda_display))

import matplotlib.pyplot as plt
from wordcloud import WordCloud 

#create_WordCloud(result['Lemma_text'].loc[result.Topic == 'type 1'], title="Most used words in topic 1")




plt.subplot(2 , 2 , 1)
plt.imshow(WordCloud(width = 500, height = 500,
                      background_color ='black',
                      min_font_size = 15).fit_words(dict(lda_model.show_topic(topicid=1, topn=40))))
plt.xticks([])
plt.yticks([])
plt.title("Most used words in LDA topic"+str(1))
plt.subplot(2 , 2 , 2)
plt.imshow(WordCloud(width = 500, height = 500,
                      background_color ='black',
                      min_font_size = 15).fit_words(dict(lda_model.show_topic(topicid=2, topn=40))))
plt.xticks([])
plt.yticks([])
plt.title("Most used words in LDA topic"+str(2))
plt.subplot(2 , 2 , 3)
plt.imshow(WordCloud(width = 500, height = 500,
                      background_color ='black',
                      min_font_size = 15).fit_words(dict(lda_model.show_topic(topicid=3, topn=40))))
plt.xticks([])
plt.yticks([])
plt.title("Most used words in LDA topic"+str(3))
plt.subplot(2 , 2 , 4)
plt.imshow(WordCloud(width = 500, height = 500,
                      background_color ='black',
                      min_font_size = 15).fit_words(dict(lda_model.show_topic(topicid=4, topn=40))))
plt.xticks([])
plt.yticks([])
plt.title("Most used words in LDA topic"+str(4))

plt.show()




###############################################################

'HDP'

print('Next part is HDP')

import tomotopy as tp
term_weight = tp.TermWeight.ONE
hdp = tp.HDPModel(tw=term_weight,min_cf=5,rm_top=7,gamma=1,alpha=0.1,initial_k=10,seed=99999)


for vec in texts:
    hdp.add_doc(vec)

hdp.burn_in =100
hdp.train(0)

print('Num docs:', len(hdp.docs),',Vocab size:',hdp.num_vocabs,
    ',Num words:',hdp.num_words)

print("Removed top words:",hdp.removed_top_words)

# Train Model

for i in range(0,1000,100):
    hdp.train(100)
    print('Iteration:{}\tLog-likelihood:{}\tNum of topics: {}'.format(i,hdp.ll_per_word,hdp.live_k))



import matplotlib.pyplot as plt
from wordcloud import WordCloud

hdp_model = HdpModel(corpus=corpus, id2word=dictionary)
print(hdp_model.show_topics())


import pyLDAvis.gensim_models

hdp_display = pyLDAvis.gensim_models.prepare(hdp_model, corpus, dictionary, sort_topics = False)
print(pyLDAvis.display(hdp_display))


for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(WordCloud(width=500, height=500,
                         background_color='white',
                         min_font_size=15).fit_words(dict(hdp_model.show_topic(topic_id=i+1, topn=40))))
    plt.xticks([])
    plt.yticks([])
    plt.title("Freq-words in HDP topic" + str(i+1),fontsize=6)
plt.show()