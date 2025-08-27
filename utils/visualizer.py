import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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
        background_color="white",
        width=800,
        height=400,
        max_words=150
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Word Cloud for "{text_column}"', fontsize=16)
    return fig