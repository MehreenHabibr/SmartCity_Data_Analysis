import os
import PyPDF2
#import PdfReader
from PyPDF2 import PdfFileReader
import argparse
import pandas as pd
import pickle
import re
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet', quiet=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfFileReader, PdfReader
import warnings
warnings.filterwarnings("ignore")


def read_pdfs(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filename_root, filename_ext = os.path.splitext(filename)
            if '_0' in filename_root or filename_root.endswith('AZ'):
                filename_root = filename_root.replace('_0', '').replace('AZ', '')
            state = filename_root[:2]
            city = filename_root[3:]
            city_name = str(city + "," + state)
            with open(os.path.join(folder_path, filename), "rb") as f:
                pdf_reader = PdfReader(f)
                page_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text += page.extract_text()
                data.append((city_name, page_text))
                
    return data


    
def create_df(data):
    df = pd.DataFrame(data, columns=['city', 'raw text'])
    pd.set_option('display.max_rows', None)
    return df


def read_df(df_path):
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    return df


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


def vectorize_data(clean_text):
    tfidf = TfidfVectorizer(lowercase=False)
    X = tfidf.fit_transform(clean_text)
    return X, tfidf

def append_newdoc(directory, document):
    if document.endswith(".pdf"):
        document_path = os.path.join(directory, document)
        base_filename = os.path.basename(document_path)
        filename = base_filename.split('/')[-1]
        filename_root, filename_ext = os.path.splitext(filename)
        state = filename_root[:2]
        city = filename_root[3:]
        city_name = str(city + "," + state)
        with open(document_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            raw_doc = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                raw_doc +=page.extract_text()
                                                              
    return city_name, raw_doc



parser = argparse.ArgumentParser(description="Track which pdf is put into terminal")
parser.add_argument("--document", help="pdf passed into cmd", required=True)
args = parser.parse_args()

def main(args):
    df_path = 'smartcity_df.pickle'
    if os.path.exists(df_path):
        df = read_df(df_path)
    else:
        data = read_pdfs('smartcity/')
        df = create_df(data)
        with open(df_path, 'wb') as f:
            pickle.dump(df, f)
    city_name, raw_doc = append_newdoc('smartcity', args.document)
    df = pd.concat([df, pd.DataFrame({'city': city_name, 'raw text': raw_doc}, index=[0])], ignore_index=True)
    df = clean_pdfs(df)
    if not os.path.isfile('model.pkl'):
        with open('model.pkl', 'wb') as f:
            pickle.dump(hac_model, f)
    else:
        with open('model.pkl', 'rb') as f:
            Hac_model = pickle.load(f)
    X, tfidf = vectorize_data(df['clean text'].tolist())
    cluster_ids = Hac_model.fit_predict(X.toarray())
    df['cluster id'] = cluster_ids
    new_doc_index = df[df['city'] == city_name].index[0]
    new_doc_cluster_id = df.loc[new_doc_index, 'cluster id']
    print(f"[{city_name}] clusterid: {new_doc_cluster_id}")
    df.to_csv('smartcity_predict.tsv', sep='\t', escapechar='\\')
    print(f"DataFrame append to an output file (smartcity_predict.tsv)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster SmartCity PDFs')
    parser.add_argument('--document', required=True, type=str, help='PDF document to be added and clustered')
    args = parser.parse_args()
    # remove "smartcity/" directory from document name
    args.document = args.document.split("/")[-1]
    main(args)

                                                              
                                                              
                                                              
                                                              
                      
    
            
                      
                      
                      
                      
                      
                      
                      
                      
                
