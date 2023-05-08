
#from ..project3 import *
import sqlite3
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

def create_df(data):
    df = pd.DataFrame(data, columns=['city', 'raw text'])
    pd.set_option('display.max_rows', None)
    return df


def test_create_df():
    data = {'city': ['New York', 'Chicago', 'Los Angeles'],
            'raw text': ['This is New York', 'Welcome to Chicago', 'Greetings from Los Angeles']}
    expected_columns = ['city', 'raw text']
    expected_data = pd.DataFrame(data, columns=expected_columns)

    # Call the function
    result = create_df(data)

    # Check the output
    assert isinstance(result, pd.DataFrame)
    assert result.equals(expected_data)
    assert result.columns.tolist() == expected_columns
    assert result.shape == expected_data.shape


