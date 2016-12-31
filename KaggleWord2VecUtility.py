import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
import re

class KaggleWord2VecUtility(object):
	@staticmethod
	def reviewto_wordlist(review,remove_stopwords=False):
		review_text=BeautifulSoup(review,"lxml").get_text()
		review_text=re.sub("[^a-zA-Z]"," ",review_text)
		words=review_text.lower().split()
		if remove_stopwords:
			stops=set(stopwords.words("english"))
			words=[w for w in words if not w in stops]
		return (words)
	@staticmethod
	def review_to_sentences(review,tokenizer,remove_stopwords=False):
		raw_sentences=tokenizer.tokenize(review.decode('utf8').strip())
		sentences=[]
		for raw_sentence in raw_sentences:
			if len(raw_sentence)>0:
				sentences.append(KaggleWord2VecUtility.reviewto_wordlist(raw_sentence,remove_stopwords))
		return sentences
