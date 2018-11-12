from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

def preprocess_bio_data(data):
	stop_words = set(stopwords.words('english')) 
	result = []
	lemmatizer = WordNetLemmatizer()
	for sentence in data:
		sen = []
		for item in sentence:
#             if item['token'] not in stop_words:
			# if item['label']=='I':
			#     tup = (lemmatizer.lemmatize(item['token']),item['pos'],'B')
			# else:
			tup = (lemmatizer.lemmatize(item['token']),item['pos'],item['label'])
			sen.append(tup)
		result.append(sen)
	return result

def preprocess_sentence(data):
	lemmatizer = WordNetLemmatizer()
	tokens = word_tokenize(data)
	for token in tokens:
		token = lemmatizer.lemmatize(token)
	return pos_tag(tokens)
