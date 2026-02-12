import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np

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

def plot_model_comparison(results_list):
    df = pd.DataFrame(results_list)
    df_melted = df.melt(id_vars='model_name', value_vars=['accuracy', 'f1_score'], 
                        var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x='model_name', y='Score', hue='Metric', palette='magma', ax=ax)
    ax.set_title("Model Performance Comparison", fontsize=16)
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=15)
    return fig

def plot_feature_importance(model, vectorizer, top_n=20):
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, 'coef_'):
        if model.coef_.shape[0] == 1:
            importances = model.coef_[0]
        else:
            importances = np.mean(np.abs(model.coef_), axis=0)
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title(f"Top {top_n} Most Significant Features", fontsize=16)
    ax.set_xlabel("Importance Magnitude")
    return fig

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.set_title("Learning Curve", fontsize=16)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    ax.grid(True)
    return fig