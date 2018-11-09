import pandas as pd
import nltk
import csv

def generate_word_list_json():
    df = pd.read_csv('review_text.csv',encoding='utf8')
    docs = df['reviews'].values
    data = []
    for doc in docs:
        tokens = nltk.word_tokenize(doc)
        pos = nltk.pos_tag(tokens)
        sentence = []
        for item in pos:
            words = {}
            words['token'] = item[0]
            words['pos'] = item[1]
            words['label'] = 'O' #Default BIO tag
            sentence.append(words)
        data.append(sentence)
    import json
    with open('reviews-aspect', 'w') as outfile:
        json.dump(data, outfile)