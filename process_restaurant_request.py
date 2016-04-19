# -*- coding: utf-8 -*-

import json
import re
import sys
import os
import time
import urllib
import pickle
import foursquare
import numpy as np
from bs4 import BeautifulSoup
from gensim.models import word2vec
from sklearn.externals import joblib

reload(sys)
sys.setdefaultencoding('utf-8')

from pymystem3 import Mystem

m = Mystem()


def collect_vocabs():
    # Getting words from all our vocabularies with '.lst' extensions:
    vocabulary = {}
    voc_files = [f for f in os.listdir('vocabularies/') if f.endswith('.lst')]
    for f in voc_files:
        words = set()
        for line in open('vocabularies/' + f, 'r'):
            lemma = line.strip().split(': ')[0]
            words.add(lemma.strip())
        voc_name = f.replace('.lst', '')
        vocabulary[voc_name] = words
    return vocabulary


###############################################################################

# start = time.time()
vocabulary = collect_vocabs()
food_model = joblib.load("models/food.pkl")
service_model = joblib.load("models/service.pkl")
interior_model = joblib.load("models/interior.pkl")
food_model = pickle.loads(food_model)
service_model = pickle.loads(service_model)
interior_model = pickle.loads(interior_model)
w2v_model = word2vec.Word2Vec.load_word2vec_format('models/webcorpora.model.bin', binary=True)
# print 'All models successfully loaded! it took ', time.time() - start, " seconds."

###############################################################################


def vocab_check(reviews, vocabulary):
    # Checking for vocabulary words in the review
    vocab_vectors = list()
    counter = 1
    for review in reviews:
        vector = []
        size = len(review)
        if size > 0:
            for voc in vocabulary:
                voc_words = [w for w in review if w.split('_')[0] in vocabulary[voc]]
                value = len(voc_words) / float(size)
                vector.append(value)
            counter += 1
            vocab_vectors.append(vector)
    vocab_vectors = np.array(vocab_vectors)
    return vocab_vectors


def foursquare_crawl(name):
    client = foursquare.Foursquare(client_id='3XM4S22O2C4HRHW1WSGIVMMNJZPMMYS3YYUFNGS1BNBNJMDY', \
                                   client_secret='3L3MP0TQBPN05KPMM2AL0DO2L5XNL1AP30TCGDR4GQEKBLXW')
    place = client.venues.search(params={'near': 'Moscow', 'limit': 10, 'query': name})
    tr1 = json.dumps(place)
    tr2 = json.loads(tr1)
    reviews = []
    for i in tr2['venues']:
        cafeid = i['id']
        e = client.venues.tips(cafeid)
        pop1 = json.dumps(e)
        pop2 = json.loads(pop1)
        for tip in pop2['tips']['items']:
            text = tip['text']
            reviews.append(text)
    return reviews


def zoon_crawl(name, reviews):
    respData = urllib.urlopen("http://zoon.ru/search/?query=" + name).read()
    rutext = respData.decode("utf-8")
    soup1 = BeautifulSoup(rutext, 'html.parser')
    links = soup1.find_all('li', 'service-item pd20 js-results-item  ')
    for link in links:
        if 'restaurant' in str(link):
            url = re.findall(r'href="(.*?)"', str(link))[1]
            x = urllib.urlopen(url).read()
            soup = BeautifulSoup(x, 'html.parser')
            texts = soup.find_all('div', 'simple-text comment-text js-comment-text')
            for text in texts:
                reviews.append(text.get_text())
    return reviews


def mystem(sentence):
    # Preprocess reviews
    lemmas = m.analyze(sentence)
    lemmas_with_pos = []
    for i in lemmas:
        if 'analysis' in i.keys():
            if len(i['analysis']) != 0:
                l = i['analysis'][0]['lex']
                pos = i['analysis'][0]['gr'].split(',')[0].split('=')[0]
                lemmas_with_pos.append(l + '_' + pos)
    return lemmas_with_pos


def preprocess(reviews):
    # returns list of lists
    processed = []
    for review in reviews:
        review = mystem(review)
        processed.append(review)
    return processed


def classify(reviews, vocab_vectors, model):
    matrix = []
    for review in reviews:
        for lemma in review:
            try:
                matrix.append(w2v_model[lemma])
            except KeyError:
                pass
    matrix = np.mean(np.array(matrix), axis=0)
    vocab_vectors = np.mean(vocab_vectors, axis=0)
    result = np.concatenate((matrix, vocab_vectors))
    sentiment = model.predict(result)
    return sentiment


def main(name):
    # start = time.time()
    reviews = foursquare_crawl(name)
    reviews = zoon_crawl(name, reviews)
    processed = preprocess(reviews)
    # print 'Number of reviews: ', len(processed)
    # print 'All reviews crawled. It took ' , time.time() - start, " seconds. Start model analysing..."
    vocab_vectors = vocab_check(processed, vocabulary)
    food_sentiment = classify(processed, vocab_vectors, food_model)
    service_sentiment = classify(processed, vocab_vectors, service_model)
    interior_sentiment = classify(processed, vocab_vectors, interior_model)
    # print "Sentiment analysis done! It took ", time.time() - start, " seconds."
    return {'food_sentiment': food_sentiment[0], 'service_sentiment': service_sentiment[0], \
            'interior_sentiment': interior_sentiment[0]}


if __name__ == '__main__':
    name = sys.argv[1]
    processed = main(name)
    print processed

