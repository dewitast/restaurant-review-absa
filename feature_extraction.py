import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

file = pd.read_csv("reviews.csv")

reviews = file['reviews'].values.tolist()
foods = file['food'].values.tolist()
services = file['service'].values.tolist()
prices = file['price'].values.tolist()
places = file['place'].values.tolist()

tokenize_reviews = []
for review in reviews:
    word_tokens = word_tokenize(review)
    tokenize_reviews.append(word_tokens)
    
stoplist = set(stopwords.words('english'))
filtered_reviews = []
words = []

def RemovePunctAndStopWords(tokens):
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokens if nonPunct.match(w)]
    #Remove the stopwords from filtered text
    filtered_words = [word for word in filtered if word.lower() not in stoplist]
    frequent_words=['the','and','of','this','am','etc','also','are','were','was','is']
    filtered_words = [word for word in filtered_words if word.lower() not in frequent_words]
    filtered_words=[word.lower() for word in filtered_words]
    return filtered_words

for review in tokenize_reviews:
    review = RemovePunctAndStopWords(review)
    filtered_reviews.append(review)

def ngram_list(word_list, n):
    all_ngrams = list(ngrams(word_list, n))
    ngram_res = []
    for ngram in all_ngrams:
        ngram_res.append(ngram)
    return ngram_res

trigram_result = []
# for i in range(len(filtered_reviews)):
#     if (len(filtered_reviews[i]) > 2):
#         trigram = ngram_list(filtered_reviews[i], 3)
#         trigram_result.append(dict(Counter(trigram)))

for i in range(len(filtered_reviews)):
    unigram = ngram_list(filtered_reviews[i], 1)
    trigram_result.append(dict(Counter(unigram)))
#     bigram = {}
#     if (len(filtered_reviews[i]) > 1):
#         bigram = ngram_list(filtered_reviews[i], 2)
#     trigram_result.append({**dict(Counter(unigram)), **dict(Counter(bigram))})

trigram_set = set()
for trigram in trigram_result:
    for key in trigram:
        trigram_set.add(key)

output_feature = pd.DataFrame(0, columns = trigram_set, index = [i for i in range(len(filtered_reviews))])
for i, trigram in enumerate(trigram_result):
    for k, v in iter(trigram.items()):
        output_feature.at[i,k] += v

output_feature.to_csv('output_encode.csv', index=False, header=False)

food_df = pd.DataFrame(foods)
service_df = pd.DataFrame(services)
price_df = pd.DataFrame(prices)
place_df = pd.DataFrame(places)

def predictData(datas, aspect):
    if (aspect == "food"):
        classifier = joblib.load('random_forest_food.pkl')
    if (aspect == "price"):
      classifier = joblib.load('random_forest_price.pkl')
    if (aspect == "place"):
      classifier = joblib.load('random_forest_place.pkl')
    if (aspect == "service"):
      classifier = joblib.load('random_forest_service.pkl')
    
    tokenize_data = []
    filtered_data = []
    for review in datas:
        data_tokens = word_tokenize(review)
        tokenize_data.append(data_tokens)
    
    for review in tokenize_data:
        review = RemovePunctAndStopWords(review)
        filtered_data.append(review)
    
    data_result = []
    for i in range(len(filtered_data)):
        data = ngram_list(filtered_data[i], 1)
        data_result.append(data)

    print(data_result)

    feature = []
    for trigram in trigram_set:
        feature.append(data_result[0].count(trigram) if trigram in data_result[0] else 0)
                
    output_data = pd.DataFrame(feature)
    
    result = classifier.predict(output_data.T)
    result_proba = classifier.predict_proba(output_data.T)
    print(result_proba)
    if result > 0:
        return('positive')
    elif result < 0:
        return('negative')
    else:
        return('neutral')

text = ["Call me nitpicky, but overcooked yolks pretty much kills the dish."]

predictData(text, "food")
predictData(text, "price")
predictData(text, "place")
predictData(text, "service")