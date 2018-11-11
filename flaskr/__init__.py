import os

from flask import Flask,redirect, url_for, request, render_template
from aspect_extractor import *

app = Flask(__name__)
@app.route("/",methods = ['GET','POST'])
def index():
	if request.method == 'POST':
		review = request.form['review']
		aspect_extractor = AspectExtractor()
		bio,aspects = aspect_extractor.extract_aspect(review)
		return render_template('index.html', review = review, bio=bio, aspects = aspects)
	else:
		return render_template('index.html')

	

# @app.route('/result',methods = ['POST'])
# def result():
#     if request.method == 'POST':
#         review = request.form['review']
#         aspect_extractor = AspectExtractor()
#         bio,aspects = aspect_extractor.extract_aspect(review)

#     return render_template('result.html', review = review, bio=bio, aspects = aspects)
