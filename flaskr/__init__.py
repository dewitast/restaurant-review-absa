import os

from flask import Flask,redirect, url_for, request, render_template
from aspect_extractor import *

app = Flask(__name__)
@app.route("/")
def index():
   return render_template("index.html")

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        review = request.form['review']
        aspect_extractor = AspectExtractor()
        aspect = aspect_extractor.extract_aspect(review).tolist()

    return render_template('result.html', review = review, aspect = aspect)



# if __name__ == '__main__':
#    app.run(debug = True)