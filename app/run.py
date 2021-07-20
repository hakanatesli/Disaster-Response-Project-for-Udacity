import json
import plotly
import pandas as pd
import re

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine

import random
from collections import Counter

app = Flask(__name__)

def tokenize(text):
     #Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize
    words = nltk.word_tokenize(text)
    
    #Remove stopwords
    words= [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
  
    clean_tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(word)) for word in words]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    #genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #top 10 categories
    category_counts= df.drop(['id','message','original','genre'], axis=1).sum().sort_values(ascending=False)
    category_names= list(category_counts.index)[0:10]
    
    #social media categories
    social = df[df['genre'] == 'social']   
    social_counts = social.drop(['id','message','original','genre'], axis=1).sum().sort_values(ascending=False)
    social_cat = list(social_counts.index)[0:10]
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
            {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=social_cat,
                    y=social_counts
                )
            ],

            'layout': {
                'title': 'Social Media - Distribution of top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()