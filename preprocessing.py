#LIBRAIRIES
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
#METHODS

def remove_tags(texts:str):
    '''remove tags'''
    return re.sub(r'<[^>]+>', r'', texts)

def translate(s):
    '''remove punctuation signs'''
    punc = string.punctuation
    dict_punc = {p : '' for p in punc}
    ttab = str.maketrans(dict_punc)
    return s.translate(ttab)

def to_lowercase(s):
    '''lowercase a string'''
    return s.lower()

def tokenize(texts:str ):
    '''tokenize a string'''
    tokens_list = word_tokenize(texts) #tokenization
    return tokens_list

def lemmatizer(token_list:list):
    wnl = WordNetLemmatizer()
    return {wnl.lemmatize(token) for token in token_list}

def remove_stop_words(tokens_list:set):
    '''remove stop_words'''
    stop_words_list = stopwords.words('english')
    non_stop_words_tokens = [token for token in tokens_list if token not in stop_words_list]
    return ' '.join(tokens_list)

def clean_text (s:str, lemmatize = False):
    s = to_lowercase(remove_tags(s)) #remove tags then lowercase the string
    s = translate(s) #remove punctuation signs
    if lemmatize:
        s = remove_stop_words(lemmatizer(tokenize(s)))
    else:
        s = remove_stop_words(tokenize(s))
    return s
    
    

    