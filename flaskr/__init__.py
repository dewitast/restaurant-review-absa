import os

from flask import Flask,redirect, url_for, request, render_template
from aspect_extractor import *
from aspect_classifier import get_aspects
from preprocess import preprocess_sentence, convert_bio
from feature_extraction import predictData

app = Flask(__name__)
@app.route("/",methods = ['GET','POST'])
def index():
	if request.method == 'POST':
		review = request.form['review']
		data = preprocess_sentence(review)
		aspect_extractor = AspectExtractor()
		bio, aspect_terms = aspect_extractor.extract_aspect(data, review)
		bio = convert_bio(bio)
		aspects = get_aspects(data, bio)
		aspect_map = {}
		for i in range(len(aspects)):
			aspect_map[aspect_terms[i]] = aspects[i]
		sentiment_food = predictData([review], "food")
		sentiment_price = predictData([review], "price")
		sentiment_place = predictData([review], "place")
		sentiment_service = predictData([review], "service")
		return render_template('index.html', review = review, bio=bio, aspect_terms = aspect_terms, aspects = aspect_map, food = sentiment_food, price = sentiment_price, place = sentiment_place, service = sentiment_service)
	else:
		return render_template('index.html')

	

# @app.route('/result',methods = ['POST'])
# def result():
#     if request.method == 'POST':
#         review = request.form['review']
#         aspect_extractor = AspectExtractor()
#         bio,aspects = aspect_extractor.extract_aspect(review)

#     return render_template('result.html', review = review, bio=bio, aspects = aspects)
