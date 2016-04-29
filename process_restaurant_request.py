#!/usr/bin/anaconda
# -*- coding: utf-8 -*-

import json
import re
import sys
import os
import time
import urllib
import pickle
import zipfile
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
		voc_name = f.replace('.lst','')
		vocabulary[voc_name] = words
	return vocabulary

def vocab_check(review, vocabulary):
	# Checking for vocabulary words in the review
	vector = []
	size = len(review)
	if size > 0:
		for voc in vocabulary:
			voc_words = [w for w in review if w.split('_')[0] in vocabulary[voc]]
			value = len(voc_words)/float(size)
			vector.append(value)
	return vector

def foursquare_crawl(name):
	client = foursquare.Foursquare(client_id='3XM4S22O2C4HRHW1WSGIVMMNJZPMMYS3YYUFNGS1BNBNJMDY', \
								   client_secret='3L3MP0TQBPN05KPMM2AL0DO2L5XNL1AP30TCGDR4GQEKBLXW')
	place=client.venues.search(params={'near': 'Moscow', 'limit': 10, 'query': name})
	tr1=json.dumps(place)
	tr2=json.loads(tr1)
	reviews = []
	for i in tr2['venues']:
		cafeid=i['id']
		e=client.venues.tips(cafeid)
		pop1=json.dumps(e)
		pop2=json.loads(pop1)
		for tip in pop2['tips']['items']:
			text=tip['text']
			reviews.append(text)
	return reviews
		
def zoon_crawl(name, reviews):
	name = name.encode('utf8')
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
				lemmas_with_pos.append(l+'_'+pos)
	return lemmas_with_pos

def preprocess(reviews):
	# returns list of lists
	processed = []
	for review in reviews:
		review = mystem(review)
		processed.append(review)
	return processed

def classify(reviews, model, vocabulary):
	sentiments = []
	for review in reviews:
		vocab_vector = vocab_check(review, vocabulary)
		matrix = list()
		for lemma in review:
			try:
				matrix.append(w2v_model[lemma])
			except KeyError: pass
		matrix = np.mean(np.array(matrix), axis = 0)
		vocab_vector = np.array(vocab_vector)
		try:
			result = np.concatenate((matrix, vocab_vector))
			sentiment = model.predict(result)[0]
			if sentiment == 'positive':
				sentiments.append(1)
			elif sentiment == 'negative':
				sentiments.append(-1)
			else:
				sentiments.append(0)
		except ValueError: continue
	return sentiments_mean(sentiments)

def sentiments_mean(sentiments):
	final = np.mean(np.array(sentiments))
	# print sentiments
	if final > 0: return 'Отлично'
	elif final < 0: return 'negative'
	else: return 'neutral'

def main(name):
	start = time.time()
	reviews = foursquare_crawl(name)
	reviews = zoon_crawl(name, reviews)
	if len(reviews) == 0:
		print "No reviews found! Probably, there is no such restaurant name"
		return {'food_sentiment':'neutral', 'service_sentiment':'neutral',\
				'interior_sentiment':'neutral', 'reviews_number':0, 'reviews':[]}
	else:
		processed = preprocess(reviews)
		food_sentiment = classify(processed, food_model, vocabulary)
		service_sentiment = classify(processed, service_model, vocabulary)
		interior_sentiment = classify(processed, interior_model, vocabulary)
		return {'food_sentiment':food_sentiment, 'service_sentiment':service_sentiment,\
				'interior_sentiment':interior_sentiment, 'reviews_number':len(processed),\
				'reviews':reviews[:5]}

###############################################################################

vocabulary = collect_vocabs()
food_model = joblib.load("models/food.pkl")  #'sklearn.svm.classes.SVC'
service_model = joblib.load("models/service.pkl")
interior_model = joblib.load("models/interior.pkl")
food_model = pickle.loads(food_model)
service_model = pickle.loads(service_model)
interior_model = pickle.loads(interior_model)

w2v_model = word2vec.Word2Vec.load_word2vec_format('models/webcorpora.model.bin', binary= True)

if __name__ == "__main__":
	name = sys.argv[1].decode('utf8')
	processed = main(name)
	print processed
