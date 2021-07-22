"""
TR: Bu dosya ile aşağıdaki işlemler yapılmaktadır:
    -Veri Yükleme,
    -Veriyi Temizleme,
    -Veriyi DB'ye kayıt etme.

EN: The following operations are performed with this file:
    -Data Upload,
    -Cleaning Data,
    -Saving the data to the DB.
"""
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    TR: Veriyi yükleme fonksiyonudur.
        
        GİRDİ: -messages_filepath: Mesajlar dosyası yolu
               -categories_filepath: Kategoriler dosyası yolu
               
        ÇIKTI: Mesajlar ve Kategoriler dosyalarının birleştirilmiş hali
    
    
    EN: The function of loading data.
        
        INPUT: -messages_filepath: Path to Messages file
               -categories_filepath: Path to categories file
               
        OUTPUT: Merged Messages and Categories files    
    """
    messages = pd.read_csv(messages_filepath, delimiter=",", encoding = 'utf-8')
    categories = pd.read_csv(categories_filepath,  delimiter=",", encoding = 'utf-8')
    df = messages.merge(categories, how='inner', on= 'id')
    return df


def clean_data(df):
    """
    TR: Veriyi temizler.
        
        GİRDİ: Kirli DataFrame
        
        ÇIKTI: Temizlenmiş DataFrame
        
    EN: Cleaned the data.
        
         INPUT: Dirty DataFrame
        
         OUTPUT: Cleaned DataFrame
    """
    #Separates the categories data with a semicolon. 
    #Sets the first part of the allocated data to be the header and the second part to be data.
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0]
    categories.columns = row.apply(lambda x: x[:-2])
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x:1 if int(x[-1]) > 1 else int(x[-1]))
    #Drop old categories column
    df.drop('categories', axis=1, inplace=True)
    #Add new categories columns
    df = pd.concat([df, categories], axis=1)
    #Removed Duplicated data, You can choose one of two options.
    df = df[df.duplicated()==False]
    #df.drop_duplicates(inplace=True)

    df = df[~(df.isnull().any(axis=1))|(df.original.isnull())]
    
    return df


def save_data(df, database_filename):
    """
    TR: Temizlenmiş veriyi DB'ye kayıt eder.
        
        GİRDİ: df: Temizlenmiş DataFrame
               database_filename: VeriTabanına verilecek isim
        
        ÇIKTI: Çıktı Yok.
        
    EN: Saves the cleaned data to the DB.
        
         INPUT: df: Cleaned DataFrame
                database_filename: The name to be given to the database
        
         OUTPUT: No Output.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
