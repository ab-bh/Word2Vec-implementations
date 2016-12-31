#loding all required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

#loading test and train data
print "loading data..."
if __name__=='__main__':
	train=pd.read_csv('labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
	test=pd.read_csv('testData.tsv',header=0,delimiter='\t',quoting=3)
	unlabeled_train=pd.read_csv('unlabeledTrainData.tsv',header=0,delimiter='\t',quoting=3)

#word2vec
print "creating word vectors..."

clean_train_reviews=[]
for i in xrange(len(train["review"])):
	clean_train_reviews.append(" ".join(KaggleWord2VecUtility.reviewto_wordlist(train["review"][i],True)))

unlabeled_clean_train_reviews = []
for review in unlabeled_train['review']:
    unlabeled_clean_train_reviews.append( " ".join( KaggleWord2VecUtility.reviewto_wordlist( review )))

#create Bag of Words
print "creating a vector..."
vector=TfidfVectorizer(analyzer="word",max_features=50000,sublinear_tf=True,stop_words = 'english',ngram_range=(1, 2), use_idf=1,smooth_idf=1,strip_accents='unicode',min_df=3)

#tokenizing the vectors
print "tokenizing the vector..." 
vector=vector.fit(clean_train_reviews+unlabeled_clean_train_reviews)
train_data=vector.transform(clean_train_reviews)


y=train["sentiment"]

#splitting train data for testing purposes
print "splitting training data for testing purposes..."
X_train,X_test,y_train,y_test=train_test_split(train_data,y,test_size=0.2,random_state=42)


showdown=False
op=True

#showdown(removed Gaussian as performed poorly)
if showdown:
	print "classifier showdown..."
	classifiers=[
				RandomForestClassifier(n_estimators=150),
				MultinomialNB(alpha=0.0001),	
				SGDClassifier(loss='modified_huber',warm_start="True"),
				LogisticRegression(penalty="l2",C=1)
				]
	count=0
	for clf in classifiers:
		count+=1
		print "training...",count
		clf.fit(X_train,y_train)
		print "testing...",count		
		y_pred=clf.predict(X_test)
		print "result...",count,":",accuracy_score(y_test,y_pred)
if op:
	print "training classifier"
	clf=LogisticRegression(penalty="l2",C=1) #performing better than others
	clf.fit(train_data,y)

	print "training complete"

	clean_test_reviews=[]
	print "creating test data"
	for i in xrange(len(test["review"])):
		clean_test_reviews.append(" ".join(KaggleWord2VecUtility.reviewto_wordlist(test["review"][i],True)))
	test_data=vector.transform(clean_test_reviews)

	print "testing..."
	y_pred=clf.predict_proba(test_data)[:,1]
	print "testing complete"
	print "preparing submission file"
	submission=pd.DataFrame(data={"id":test['id'],"sentiment":y_pred})
	submission.to_csv('submission.csv',quoting=3,index=False)