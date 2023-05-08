import pytest
#from project3 import vectorize_data
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest
from io import StringIO
import PyPDF2
#from project0 import project0
from os import path
#from ../..project0 import project0
import sys
sys.path.append(path.abspath('../project3'))
#from cs5493sp23-project0.project
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
def clean_pdfs(df):
    mask = df['city'].str.contains('Toledo|Moreno Valley|Lubbock|Reno| Tallahassee')
    df = df.loc[~mask]
    city_seps = []
    for city in df['city']:
        city_sep = city.split(',')[0].lower()
        city_seps.append(city_sep)

    lemmatizer = WordNetLemmatizer()
    common_words = set(stopwords.words('english')) | set(['smart', 'page', 'city', 'challenge', 'contents', 'usdot', 'depatment', 'submitted', 'application', 'beyond', 'traffic', 'proposal', 'dtfh6ll6ra0000', '^', '~', 'february', '2016']) | set(city_seps)
    df.loc[:, 'clean text'] = df['raw text'].apply(lambda x: x.lower())
   # df.loc[:, 'clean text'] = df.loc[:, 'raw text'].apply(lambda x: x.lower())

    df.loc[:, 'clean text'] = df['clean text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df.loc[:, 'clean text'] = df['clean text'].apply(lambda x: word_tokenize(x))
    df.loc[:, 'clean text'] = df['clean text'].apply(lambda x: [token for token in x if token not in common_words])
    df.loc[:, 'clean text'] = df['clean text'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x if len(token) > 2 and not bool(re.search(r'\d', token))])
    df.loc[:, 'clean text'] = df['clean text'].apply(lambda x: " ".join(x))
    pd.set_option('display.max_colwidth', 100)
    return df

def test_clean_pdfs():
    # create a test DataFrame
    df = pd.DataFrame({'city': ['New York', 'Tallahassee', 'Lubbock'],
                       'raw text': ['Sample text 1.', 'Sample text 2.', 'Sample text 3.']})

    # call the function
    cleaned_df = clean_pdfs(df)

    # check if the returned object is a DataFrame
    assert isinstance(cleaned_df, pd.DataFrame)

    # check if the DataFrame has expected number of rows and columns
#    assert cleaned_df.shape == (1, 3)

    # check if the city column contains the expected value
    assert cleaned_df['city'].values[0] == 'New York'

    # check if the raw text column has been cleaned as expected
    assert cleaned_df['clean text'].values[0] == 'sample text'

