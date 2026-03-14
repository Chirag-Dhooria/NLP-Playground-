import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def _to_lowercase(text):
    return str(text).lower()

def _remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def _remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def _lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def _stem_text(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in words])

def preprocess_text(df, text_columns, options, is_qa=False, context_col=None, question_col=None):
    if is_qa:
        df['processed_context'] = df[context_col].astype(str)
        df['processed_question'] = df[question_col].astype(str)
        cols_to_process = ['processed_context', 'processed_question']
    else:
        df['processed_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
        cols_to_process = ['processed_text']

    for col in cols_to_process:
        if options.get('lowercase', False):
            df[col] = df[col].apply(_to_lowercase)
        if options.get('remove_punctuation', False):
            df[col] = df[col].apply(_remove_punctuation)
        if options.get('remove_stopwords', False):
            df[col] = df[col].apply(_remove_stopwords)
        if options.get('lemmatization', False):
            df[col] = df[col].apply(_lemmatize_text)
        if options.get('stemming', False):
            df[col] = df[col].apply(_stem_text)
            
    return df