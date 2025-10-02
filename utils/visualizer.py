import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def plot_label_distribution(df, target_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=target_column, data=df, ax=ax, palette='viridis')
    ax.set_title(f'Distribution of Labels in "{target_column}"', fontsize=16)
    ax.set_xlabel(target_column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    return fig

def generate_wordcloud(df, text_column):
    text = " ".join(review for review in df[text_column].astype(str).dropna())
    wordcloud = WordCloud(
        background_color="white", width=800, height=400, max_words=150
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Word Cloud for "{text_column}"', fontsize=16)
    return fig

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    return fig

def plot_top_ngrams(df, text_column, n=2, top_k=20):
    """
    Plots the top K N-grams from a text column.
    """
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(df[text_column].astype(str))
    bag_of_words = vec.transform(df[text_column].astype(str))
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    top_df = pd.DataFrame(words_freq[:top_k], columns = ['N-gram', 'Frequency'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Frequency', y='N-gram', data=top_df, palette='plasma')
    ax.set_title(f'Top {top_k} {n}-grams', fontsize=16)
    return fig