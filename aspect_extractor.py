import numpy as np
from nltk.corpus import stopwords
import json
from sklearn.model_selection import KFold
from nltk.stem import WordNetLemmatizer
import pycrfsuite
from sklearn.metrics import classification_report
import nltk
from preprocess import preprocess_bio_data

def get_BIO_data(filename):
	with open(filename) as f:
		labeled = json.load(f)
	return preprocess_bio_data(labeled)

def word2features(doc, i):
	word = doc[i][0]
	postag = doc[i][1]

	features = [
		'bias',
		'word.lower=' + word.lower(),
		'postag=' + postag
	]
#     Fitur untuk kata non awal kalimat
	if i > 1:
		word2 = doc[i-2][0]
		postag2 = doc[i-2][1]
		features.extend([
			'-2:word.lower=' + word2.lower(),
			'-2:postag=' + postag2
		])
	if i > 0:
		word1 = doc[i-1][0]
		postag1 = doc[i-1][1]
		features.extend([
			'-1:word.lower=' + word1.lower(),
			'-1:postag=' + postag1
		])
	else:
		features.append('BOS') # Tanda awal kalimat
	# Fitur untuk kata non akhir kalimat
	if i < len(doc)-2:
		word2 = doc[i+2][0]
		postag2 = doc[i+2][1]
		features.extend([
			'+2:word.lower=' + word2.lower(),
			'+2:postag=' + postag2
		])
	if i < len(doc)-1:
		word1 = doc[i+1][0]
		postag1 = doc[i+1][1]
		features.extend([
			'+1:word.lower=' + word1.lower(),
			'+1:postag=' + postag1
		])
	else:
		features.append('EOS') #tanda akhir kalimat
	return features

def extract_features(doc):
	return [word2features(doc, i) for i in range(len(doc))]

def get_labels(doc):
	return [label for (token, postag, label) in doc]

def get_split_index(X,n_split):
	kf = KFold(n_splits=n_split)
	train_split = []
	test_split = []
	for train_index, test_index in kf.split(X):
		train_split.append(train_index)
		test_split.append(test_index)
	return train_split,test_split

def generate_model(X,y):
	trainer = pycrfsuite.Trainer(verbose=True)
	for xseq, yseq in zip(X,y):
		trainer.append(xseq, yseq)
	trainer.set_params({
		'c1': 0.1, #L1 penalty
		'c2': 0.01, #L2 penalty
		'max_iterations': 500,
		'feature.possible_transitions': True
	})
	trainer.train('crf_model/crf.model_main')
		

def generate_model_split(X,y,train_split,test_split):
	for i in range(len(train_split)):
		trainer = pycrfsuite.Trainer(verbose=True)
		X_train = [X[j] for j in train_split[i]]
		y_train = [y[j] for j in train_split[i]]
		for xseq, yseq in zip(X_train, y_train):
			trainer.append(xseq, yseq)
		trainer.set_params({
			'c1': 0.1, #L1 penalty
			'c2': 0.01, #L2 penalty
			'max_iterations': 200,
			'feature.possible_transitions': True
		})
		trainer.train('crf_model/crf.model_'+str(i))
		
class AspectExtractor:
	def __init__(self):
		pass

	def create_model(self,review_data,split_size):
		data = get_BIO_data(review_data)
		X = [extract_features(doc) for doc in data]
		y = [get_labels(doc) for doc in data]
		if split_size==1:
			generate_model(X,y)
		else:
			train_split, test_split = get_split_index(X,split_size)
			generate_model_split(X,y,train_split,test_split)
			self.X = X
			self.y = y
			self.train_split = train_split
			self.test_split = test_split


	def evaluate_model(self):
		X = self.X
		y = self.y
		train_split = self.train_split
		test_split = self.test_split
		
		labels = {"B": 0, "I": 1,"O":2}
		target_names = ['B', 'I','O']

		report = []
		for i in range(len(train_split)):
			tagger = pycrfsuite.Tagger()
			tagger.open('crf_model/crf.model_'+str(i))
			X_test = [X[j] for j in test_split[i]]
			y_test = [y[j] for j in test_split[i]]
			y_pred = [tagger.tag(xseq) for xseq in X_test]
		#     for iterr, X_test_item in enumerate(X_test):
		#         for idx,word in enumerate(X_test_item):
		#             print(word[1]+" - "+y_test[iterr][idx]+" - "+y_pred[iterr][idx])
			truth = np.array([labels[bio] for sentence in y_test for bio in sentence])
			prediction = np.array([labels[bio] for sentence in y_pred for bio in sentence])
			report.append(classification_report(truth, prediction, target_names=target_names, output_dict=True))
		print(report)
		
		f1=0
		recall = 0
		for re in report:
			f1+=re['B']['f1-score']
			recall+=re['B']['recall']
		print(f1/len(report))
		print("Rata-rata Recall untuk B")
		print(recall/len(report))

	def extract_aspect(self,review):
		data = [review]
		X = [extract_features(doc) for doc in data]
		labels = {"B": 0, "I": 1,"O":2}
		target_names = ['B', 'I','O']
		tagger = pycrfsuite.Tagger()
		tagger.open('crf_model/crf.model_main')
		y_pred = [tagger.tag(xseq) for xseq in X]
		prediction = np.array([labels[bio] for sentence in y_pred for bio in sentence])
		print(prediction)
		review_token = nltk.word_tokenize(review)
		idx=0
		aspects = []
		while idx<len(prediction):
		    print("mauk")
		    if prediction[idx]==0:
		        aspect = review_token[idx]
		        idx += 1
		        while idx<len(prediction) and prediction[idx]==1:
		            aspect+=" "+review_token[idx]
		            idx+=1
		        aspects.append(aspect)
		    idx+=1
		return prediction, aspects