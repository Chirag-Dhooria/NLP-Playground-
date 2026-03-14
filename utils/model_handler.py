from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
import streamlit as st

@st.cache_resource(show_spinner=True)
def load_hf_pipeline(task, model_name):
    from transformers import pipeline
    try:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(task, model=model_name, device=device)
    except Exception as e:
        return str(e)

def get_sklearn_model(model_name, params={}):
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=42, **params)
    elif model_name == "Naive Bayes":
        return MultinomialNB(**params)
    elif model_name == "Support Vector Machine (SVM)":
        return LinearSVC(random_state=42, dual=False, **params)
    elif model_name == "Random Forest":
        return RandomForestClassifier(random_state=42, **params)
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42, **params)
    return None

def run_sklearn_pipeline(df, text_col, target_col, model_name, params, test_size):
    X = df[text_col]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = get_sklearn_model(model_name, params)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    return model, vectorizer, metrics, X_train_vec, y_train, y_test, y_pred

def run_hf_inference(df, task, model_name, text_col=None, context_col=None, question_col=None):
    hf_pipe = load_hf_pipeline(task, model_name)
    if isinstance(hf_pipe, str):
        raise Exception(f"Failed to load model: {hf_pipe}")
    results = []
    data_subset = df.head(20)
    for _, row in data_subset.iterrows():
        if task == "question-answering":
            res = hf_pipe(question=row[question_col], context=row[context_col])
            results.append(res['answer'])
        elif task == "summarization":
            res = hf_pipe(row[text_col], max_length=130, min_length=30, do_sample=False)
            results.append(res[0]['summary_text'])
        elif task == "sentiment-analysis":
            res = hf_pipe(row[text_col])
            results.append(f"{res[0]['label']} ({res[0]['score']:.2f})")
        elif task == "text-classification":
            res = hf_pipe(row[text_col])
            results.append(res[0]['label'])
    return results