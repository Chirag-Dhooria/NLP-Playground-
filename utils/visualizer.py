import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np

# ── Existing plots ────────────────────────────────────────────────────────────

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
    wordcloud = WordCloud(background_color="white", width=800, height=400, max_words=150).generate(text)
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
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_df = pd.DataFrame(words_freq[:top_k], columns=['N-gram', 'Frequency'])
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
        importances = model.coef_[0] if model.coef_.shape[0] == 1 else np.mean(np.abs(model.coef_), axis=0)
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
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    ax.plot(train_sizes, np.mean(test_scores,  axis=1), 'o-', color="g", label="Cross-validation score")
    ax.set_title("Learning Curve", fontsize=16)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    ax.grid(True)
    return fig

# ── LIME plots ────────────────────────────────────────────────────────────────

def plot_lime_explanation(explanation, label_idx=0):
    word_weights = explanation.as_list(label=label_idx)
    if not word_weights:
        return None
    words   = [w for w, _ in word_weights]
    weights = [v for _, v in word_weights]
    colors  = ['#2ecc71' if v > 0 else '#e74c3c' for v in weights]
    fig, ax = plt.subplots(figsize=(10, max(4, len(words) * 0.45)))
    ax.barh(range(len(words)), weights, color=colors, align='center')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("LIME weight  (green = supports class, red = opposes)", fontsize=11)
    ax.set_title(f"LIME — word contributions for class: '{explanation.class_names[label_idx]}'", fontsize=14)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig

def plot_lime_html(explanation, label_idx=0):
    try:
        return explanation.as_html(label=label_idx)
    except Exception:
        return None

# ── SHAP plots ────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_result, max_display=20):
    try:
        import shap
    except ImportError:
        return None
    sv = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    X_vec = shap_result["X_vec"]
    if isinstance(sv, list):
        mean_abs = [np.abs(s).mean() for s in sv]
        sv = sv[int(np.argmax(mean_abs))]
    X_dense = X_vec.toarray() if hasattr(X_vec, "toarray") else X_vec
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_dense, feature_names=feature_names,
                      max_display=max_display, show=False, plot_type="dot")
    ax = plt.gca()
    ax.set_title("SHAP Summary — global feature impact", fontsize=14)
    fig = plt.gcf()
    fig.tight_layout()
    return fig

def plot_shap_waterfall(shap_result, sample_idx=0, class_idx=0):
    try:
        import shap
    except ImportError:
        return None
    sv        = shap_result["shap_values"]
    explainer = shap_result["explainer"]
    X_vec     = shap_result["X_vec"]
    feature_names = shap_result["feature_names"]
    sv_sample = sv[class_idx][sample_idx] if isinstance(sv, list) else sv[sample_idx]
    X_dense   = X_vec.toarray() if hasattr(X_vec, "toarray") else X_vec
    base_val  = float(explainer.expected_value[class_idx]
                      if isinstance(explainer.expected_value, (list, np.ndarray))
                      else explainer.expected_value)
    exp_obj = shap.Explanation(values=sv_sample, base_values=base_val,
                               data=X_dense[sample_idx], feature_names=feature_names)
    fig, _ = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(exp_obj, max_display=15, show=False)
    fig = plt.gcf()
    class_label = shap_result["class_names"][class_idx] if class_idx < len(shap_result["class_names"]) else str(class_idx)
    fig.suptitle(f"SHAP Waterfall — sample #{sample_idx}  |  class: '{class_label}'", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig

def plot_shap_force(shap_result, sample_idx=0, class_idx=0):
    try:
        import shap
    except ImportError:
        return None
    sv        = shap_result["shap_values"]
    explainer = shap_result["explainer"]
    X_vec     = shap_result["X_vec"]
    feature_names = shap_result["feature_names"]
    sv_sample = sv[class_idx][sample_idx] if isinstance(sv, list) else sv[sample_idx]
    X_dense   = X_vec.toarray() if hasattr(X_vec, "toarray") else X_vec
    base_val  = float(explainer.expected_value[class_idx]
                      if isinstance(explainer.expected_value, (list, np.ndarray))
                      else explainer.expected_value)
    shap.initjs()
    fp = shap.force_plot(base_val, sv_sample, X_dense[sample_idx],
                         feature_names=feature_names, show=False)
    return shap.getjs() + shap.plots.force(fp, matplotlib=False).html()