**MEHREEN HABIB**
---------
> ## Project Title: CS5293, spring 2023 Project 3
### Project Description
 **This project aims to download and clean pdf documents, create and explore clustering models, perform topic modeling to derive meanings.
There is a templated project3.ipynb that we need to follow and complete. Submitted a completed verision of project3.ipynb with answers and output in the notebeook.This notebook produced an output file (smartcity_eda.tsv) and a saved model (model.pkl)
The steps to develop the executable script 
Load and clean the pdfs.
Load the best model for k optimal clusters.
Use the model to predict the cluster that the new document belongs to.
Print the output
Write/append new city in the output file as a TSV.**
 
 Command for running the project:
> pipenv run python project3.py --document "FL Miami.pdf"

Command for running the test scripts:
> pipenv run python -m pytest

 **Modules**
 1. argparse : This provides a convenient way to parse command line arguments in Python.
 2. numpy as np : This provides support for working with arrays and numerical operations
 3. pandas as pd : This provides data structures and functions for working with tabular data
 5. PyPDF2 - Utility to read and write PDFs with Python
 6. autopep8 - Tool that automatically formats Python code
 7. Pytest - Testing framework
 8. Black - Code formatter


 #### Approach to Develope the code
---
1. `read_pdfs(folder_path)`

This defines a function called read_pdfs that reads in PDF files from a specified folder path, extracts text from each page of the PDFs, and returns a list of tuples where each tuple contains the city name and the extracted text. The function filters out PDF files that do not end with ".pdf" or do not have specific file names.

2. `create_df(data):`
This code defines a function called create_df that takes in a list of tuples containing city names and extracted text and creates a pandas DataFrame with columns 'city' and 'raw text' using this data. It sets an option to display all rows in the DataFrame, and then returns the DataFrame.

3. `read_df(df_path)`
   This code defines a function called read_df that takes in a file path for a pickled pandas DataFrame and loads the DataFrame from the file using pickle.load(). It then returns the DataFrame.
4. `clean_pdfs(df)`
The clean_pdfs function cleans the text data in the 'raw text' column of a pandas DataFrame by removing stop words, punctuation, and numbers, tokenizing the text, lemmatizing the tokens, and storing the cleaned text in a new 'clean text' column. It then returns the cleaned DataFrame.
5. `vectorize_data(clean_text)`
   This code defines a function called vectorize_data that takes in a pandas Series of cleaned text and creates a TF-IDF vectorizer object. It fits the vectorizer to the text data, transforms the text data to a sparse matrix of TF-IDF features, and returns the matrix and the fitted vectorizer object.
6.  `append_newdoc(directory, document)`
 The append_newdoc function takes in a directory path and a document filename. If the document filename ends with '.pdf', the function reads the document and extracts the text. It then extracts the city and state from the filename, combines them into a single 'city_name' string, and returns a tuple of the 'city_name' and the raw document text.
7.  `main(args)`
This code clusters Smartcity PDF documents using hierarchical agglomerative clustering (HAC). The script reads in a pickled Pandas DataFrame of PDF documents (which has already undergone some cleaning), adds a new PDF document to the DataFrame, cleans it, vectorizes the text, and runs HAC on the vectorized data to assign the new document to a cluster. The resulting DataFrame is then outputted to a TSV file. The script takes a command-line argument specifying the path to the new PDF document to be added to the DataFrame.
 
 ## Tests
---
1. **`test_clean_text.py`** :  This pytest creates a test DataFrame and calls the clean_pdfs() function on it. It then checks if the returned object is a DataFrame and if the city column contains the expected value. Finally, it checks if the raw text column has been cleaned as expected.
2. **`test_create_df.py`** : This pytest tests the create_df() function which creates a pandas DataFrame from a dictionary of city names and their corresponding raw text. It checks if the output is a pandas DataFrame, has the expected column names and data, and has the expected shape.
3.  **`test_remove_specialch.py`** : This pytest checks if the remove_special_characters() function returns an empty string when given an empty string as input. This test case ensures that the function handles the edge case of empty input correctly.
4.  **`test_vectorize_data.py`** : This pytest tests the vectorize_data function which takes a list of strings as input and returns the corresponding TF-IDF vector representation of the input text. The function tests if the returned output matches the expected output by comparing the TF-IDF vectorized data and the vocabulary generated by the function with the expected values.


pipenv install
Packages required to run this project are kept in requirements.txt file which automatically installs during installation of pipenv in step 1.


##### python_version = "3.10"

##### pytest==7.2.2



## Assumptions:
---
1. Assuming that the document is .pdf only
2. multiple characters like FL Miami.pdf should be put in "" , "FL Miami.pdf"so that they are considering as single entity.



![gif](https://github.com/MehreenHabibr/cs5293sp23-project3/blob/main/ezgif.com-optimize.gif)

