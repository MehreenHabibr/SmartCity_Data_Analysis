import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_data(clean_text):
    tfidf = TfidfVectorizer(lowercase=False)
    X = tfidf.fit_transform(clean_text)
    return X, tfidf

def test_vectorize_data():
    clean_text = ["this is some sample text", "more sample text", "some more text"]
    tfidf = TfidfVectorizer(lowercase=False)
    X_expected = tfidf.fit_transform(clean_text)
    X, tfidf_actual = vectorize_data(clean_text)
    assert np.array_equal(X.toarray(), X_expected.toarray())
    assert tfidf_actual.vocabulary_ == tfidf.vocabulary_

