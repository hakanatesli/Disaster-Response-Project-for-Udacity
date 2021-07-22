"""
TR: Bu dosya ile aşağıdaki işlemler yapılmaktadır:
    -Db'den veriyi yükleme
    -DDİ için gerekli işlemlerin yapılması
    -Modelin kurulması
    -Modelin değerlendirilmesi
    -Modelin kaydedilmesi

EN: The following operations are performed with this file:
    -Loading data from db
    -Performing the necessary procedures for NLP
    -Build of the model
    -Evaluation of the model
    -Saving the model
"""
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pickle
import re


def load_data(database_filepath):
    """ 
    TR: Bu fonksiyon db'den veri yükler.
        
        GİRDİ: database_filepath: DB'nin dosya yolu
        
        ÇIKTI: X: Modelin girdi verileri
               Y: Modelin çıktı verileri
               category_names: kategori isimleri
    
    EN: This function loads data from db.
        
        INPUT: database_filepath: DB's file path
        
        OUTPUT: X: Input data of the model
                Y: Output data of the model
                category_names: category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table ('DisasterResponse', con=engine)
    category_names = df.columns[-36:]
    X = df['message']
    Y = df[category_names]
    return X,Y, category_names

def tokenize(text):
    """
    TR: Bu dosya DDİ için gereken veri temizleme işlemlerini gerçekleştirir.
        
        GİRDİ: text
        
        ÇIKTI: words: temizlenmiş kelime dizisi
    
    EN: This file performs the data cleaning operations required for NLP.
        
        INPUT: text
        
        OUTPUT: words: cleaned word string
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #Tokenize
    words = nltk.word_tokenize(text)
    #Remove stopwords
    words= [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(lemmatizer.lemmatize(word)) for word in words]
    return words


def build_model():
    """
    TR: DDİ pipeline modeli oluşturulur. Farklı parametreler ile en iyi model bulunmaya çalışılır.
        
        GİRDİ: Girdi yok.
        
        ÇIKTI: model: En iyi model
    
    EN: The NLP pipeline model is created. It is tried to find the best model with different parameters.
        
        INPUT: No input.
        
        OUTPUT: model: Best model
    """
    pipeline = Pipeline([
      ('vect', CountVectorizer(tokenizer=tokenize)),
      ('tfidf', TfidfTransformer()),
      ('clf', MultiOutputClassifier(estimator=AdaBoostClassifier()))
    ])
    #Different parameters are tested for each function.              
    parameters = {
      'vect__min_df':[1,10,50],
      'clf__estimator__learning_rate': [0.001, 0.01, 0.1],
      'tfidf__smooth_idf': [True, False]
    }
    model  = GridSearchCV(pipeline, param_grid=parameters, cv=2) 
    return model 
                  
def evaluate_model(model, X_test, Y_test, category_names):
    """
    TR: Bu fonksiyon modeli değerlendirmektedir.
    
        GİRDİ: model: sınıflandırma modeli
               X_test: X test verileri
               Y_test: Y test verileri
               category_names: kategori isimleri
        
        ÇIKTI: Çıktı yok.
    
    EN: This function evaluates the model.
    
        INPUT: model: classification model
               X_test: X test data
               Y_test: Y test data
               category_names: category names
        
        OUTPUT: No output.
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names, digits=2))

def save_model(model, model_filepath):
    """
    TR: Bu fonskiyon eğitilmiş modeli pkl dosyası olarak kaydeder.
    
        GİRDİ: model: eğitilmiş model
               model_filepath: modelin kaydedileceği dosya yolu
        
        ÇIKTI: Çıktı yok.
    
    EN: This function saves the trained model as pkl file.
    
         INPUT: model: trained model
                model_filepath: file path to save the model
        
         OUTPUT: No output.
    """
    with open(model_filepath, 'wb') as pkl_file:
                  pickle.dump(model, pkl_file)
    pkl_file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
