import string
import nltk
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def _to_lowercase(text):
    return text.lower()

def _remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def _remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def preprocess_text(df, text_column, options):
    df['processed_text'] = df[text_column]

    if options.get('lowercase', False):
        df['processed_text'] = df['processed_text'].apply(_to_lowercase)

    if options.get('remove_punctuation', False):
        df['processed_text'] = df['processed_text'].apply(_remove_punctuation)

    if options.get('remove_stopwords', False):
        df['processed_text'] = df['processed_text'].apply(_remove_stopwords)

    return df