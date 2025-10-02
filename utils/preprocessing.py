import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
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
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

def _stem_text(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

def preprocess_text(df, text_column, options):
    df['processed_text'] = df[text_column]

    if options.get('lowercase', False):
        df['processed_text'] = df['processed_text'].apply(_to_lowercase)
    if options.get('remove_punctuation', False):
        df['processed_text'] = df['processed_text'].apply(_remove_punctuation)
    if options.get('remove_stopwords', False):
        df['processed_text'] = df['processed_text'].apply(_remove_stopwords)
    if options.get('lemmatization', False):
        df['processed_text'] = df['processed_text'].apply(_lemmatize_text)
    if options.get('stemming', False):
        df['processed_text'] = df['processed_text'].apply(_stem_text)

    return df