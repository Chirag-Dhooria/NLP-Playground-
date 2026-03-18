"""
utils/explainer.py
------------------
SHAP and LIME explainability for Scikit-Learn models trained in NLP Playground.

Supported explainers per model:
  Logistic Regression   → shap.LinearExplainer  (fast)
  SVM / LinearSVC       → shap.LinearExplainer  (fast)
  Random Forest         → shap.TreeExplainer    (fast)
  Gradient Boosting     → shap.TreeExplainer    (fast)
  Naive Bayes           → shap.KernelExplainer  (slow – sampled)

LIME works on all models as a black-box wrapper.

Public API
----------
  run_lime(model, vectorizer, raw_texts, class_names, sample_idx) -> lime Explanation
  run_shap(model, vectorizer, raw_texts, class_names)             -> ShapResult(dict)
  shap_explainer_for(model, X_train_vec)                          -> shap Explainer
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# ── LIME ──────────────────────────────────────────────────────────────────────

def _lime_predict_proba(model, vectorizer, texts):
    """
    Wraps model.predict_proba / decision_function so LIME always gets
    a proper probability array regardless of model type.
    """
    X = vectorizer.transform(texts)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    # LinearSVC has no predict_proba; use decision_function + softmax
    scores = model.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def run_lime(model, vectorizer, raw_texts, class_names, sample_idx: int = 0):
    """
    Runs LIME on a single text sample.

    Parameters
    ----------
    model       : fitted sklearn classifier
    vectorizer  : fitted TfidfVectorizer
    raw_texts   : list/Series of raw (pre-vectorised) strings
    class_names : list of label strings
    sample_idx  : which row to explain

    Returns
    -------
    lime.explanation.Explanation  or  None on import/runtime error
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return None

    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    predict_fn = lambda texts: _lime_predict_proba(model, vectorizer, texts)

    text_sample = str(list(raw_texts)[sample_idx])
    explanation = explainer.explain_instance(
        text_sample,
        predict_fn,
        num_features=15,
        num_samples=300,
        top_labels=len(class_names),
    )
    return explanation


# ── SHAP ──────────────────────────────────────────────────────────────────────

def shap_explainer_for(model, X_train_vec):
    """
    Returns the most appropriate SHAP explainer for this model type.
    Uses st.cache_resource so it's only built once per session.
    """
    import shap
    model_type = type(model).__name__

    if model_type in ("LogisticRegression", "LinearSVC"):
        # masker links explainer back to training distribution
        masker = shap.maskers.Independent(X_train_vec, max_samples=100)
        return shap.LinearExplainer(model, masker)

    elif model_type in ("RandomForestClassifier", "GradientBoostingClassifier"):
        return shap.TreeExplainer(model)

    else:
        # Naive Bayes or unknown — black-box KernelExplainer on a small sample
        background = shap.sample(X_train_vec, 50)
        predict_fn = (
            model.predict_proba
            if hasattr(model, "predict_proba")
            else model.decision_function
        )
        return shap.KernelExplainer(predict_fn, background)


def run_shap(model, vectorizer, raw_texts, class_names, max_display_rows: int = 100):
    """
    Computes SHAP values for up to `max_display_rows` samples.

    Returns a dict:
    {
      "shap_values"   : np.ndarray or list  (raw SHAP output),
      "feature_names" : list[str],
      "X_vec"         : sparse matrix,
      "class_names"   : list[str],
      "model_type"    : str,
      "explainer"     : shap Explainer,
    }
    or None on failure.
    """
    try:
        import shap
    except ImportError:
        return None

    texts = list(raw_texts)[:max_display_rows]
    X_vec = vectorizer.transform(texts)

    try:
        explainer   = shap_explainer_for(model, vectorizer.transform(raw_texts))
        shap_values = explainer.shap_values(X_vec)
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        return None

    return {
        "shap_values":   shap_values,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
        "X_vec":         X_vec,
        "class_names":   class_names,
        "model_type":    type(model).__name__,
        "explainer":     explainer,
    }