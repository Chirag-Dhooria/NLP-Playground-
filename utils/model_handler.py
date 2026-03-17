from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
import streamlit as st
import requests

# ── HuggingFace task → pipeline tag mapping ────────────────────────────────────
# Maps our internal task names to the HF pipeline task string AND
# the model card "pipeline_tag" values we accept for that task.
_TASK_PIPELINE_MAP = {
    "text-classification":  {
        "pipeline_tag": "text-classification",
        "accepted_tags": ["text-classification", "sentiment-analysis"],
    },
    "summarization": {
        "pipeline_tag": "summarization",
        "accepted_tags": ["summarization"],
    },
    "question-answering": {
        "pipeline_tag": "question-answering",
        "accepted_tags": ["question-answering"],
    },
    "sentiment-analysis": {
        "pipeline_tag": "text-classification",
        "accepted_tags": ["text-classification", "sentiment-analysis"],
    },
}


# ── Custom model validator ─────────────────────────────────────────────────────

def validate_hf_model(model_id: str, task: str) -> dict:
    """
    Calls the HuggingFace Hub API to check:
      1. The model ID exists (public repo).
      2. The model's pipeline_tag is compatible with the selected task.

    Returns a dict:
      {
        "valid":   bool,
        "message": str,          # human-readable status
        "model_id": str,         # normalised model ID
        "pipeline_tag": str | None,
      }
    """
    model_id = model_id.strip().strip("/")
    if not model_id:
        return {"valid": False, "message": "Model ID cannot be empty.", "model_id": model_id, "pipeline_tag": None}

    api_url = f"https://huggingface.co/api/models/{model_id}"
    try:
        resp = requests.get(api_url, timeout=10)
    except requests.exceptions.ConnectionError:
        return {
            "valid": False,
            "message": "Could not reach HuggingFace Hub. Check your internet connection.",
            "model_id": model_id,
            "pipeline_tag": None,
        }
    except requests.exceptions.Timeout:
        return {
            "valid": False,
            "message": "HuggingFace Hub request timed out. Try again.",
            "model_id": model_id,
            "pipeline_tag": None,
        }

    if resp.status_code == 404:
        return {
            "valid": False,
            "message": f"Model `{model_id}` was not found on HuggingFace Hub. Check the model ID.",
            "model_id": model_id,
            "pipeline_tag": None,
        }
    if resp.status_code != 200:
        return {
            "valid": False,
            "message": f"HuggingFace Hub returned status {resp.status_code}.",
            "model_id": model_id,
            "pipeline_tag": None,
        }

    model_info   = resp.json()
    pipeline_tag = model_info.get("pipeline_tag", None)

    # Normalise task key
    task_key = task.lower().replace(" ", "-")
    task_cfg  = _TASK_PIPELINE_MAP.get(task_key, {})
    accepted  = task_cfg.get("accepted_tags", [])

    if pipeline_tag is None:
        # No pipeline_tag on the card — warn but allow the user to proceed
        return {
            "valid": True,
            "message": (
                f"⚠️ Model `{model_id}` exists but has no `pipeline_tag` set on its model card. "
                "It may still work — proceed with caution."
            ),
            "model_id": model_id,
            "pipeline_tag": None,
        }

    if pipeline_tag not in accepted:
        return {
            "valid": False,
            "message": (
                f"Model `{model_id}` is tagged as **{pipeline_tag}** on HuggingFace, "
                f"but the selected task requires one of: **{', '.join(accepted)}**. "
                "Please choose a compatible model or change the task."
            ),
            "model_id": model_id,
            "pipeline_tag": pipeline_tag,
        }

    return {
        "valid": True,
        "message": f"✅ `{model_id}` is compatible with **{task}** (tag: `{pipeline_tag}`).",
        "model_id": model_id,
        "pipeline_tag": pipeline_tag,
    }


# ── HuggingFace pipeline loader (cached) ──────────────────────────────────────

@st.cache_resource(show_spinner="Downloading model from HuggingFace…")
def load_hf_pipeline(task: str, model_name: str):
    """
    Loads and caches a HuggingFace pipeline.
    Works for both preset and custom model IDs.
    Returns the pipeline object on success, or a str error message on failure.
    """
    from transformers import pipeline
    try:
        device = 0 if torch.cuda.is_available() else -1
        # Map internal task names to HF pipeline task strings
        task_key    = task.lower().replace(" ", "-")
        hf_task_str = _TASK_PIPELINE_MAP.get(task_key, {}).get("pipeline_tag", task_key)
        return pipeline(hf_task_str, model=model_name, device=device)
    except OSError as e:
        return f"Model files could not be loaded: {e}"
    except Exception as e:
        return f"Pipeline error: {e}"


# ── Scikit-Learn helpers ───────────────────────────────────────────────────────

def get_sklearn_model(model_name: str, params: dict = {}):
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    vectorizer   = TfidfVectorizer()
    X_train_vec  = vectorizer.fit_transform(X_train)
    X_test_vec   = vectorizer.transform(X_test)
    model        = get_sklearn_model(model_name, params)
    model.fit(X_train_vec, y_train)
    y_pred       = model.predict(X_test_vec)
    metrics      = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
    }
    return model, vectorizer, metrics, X_train_vec, y_train, y_test, y_pred


# ── HuggingFace inference runner ───────────────────────────────────────────────

def run_hf_inference(
    df,
    task: str,
    model_name: str,
    text_col=None,
    context_col=None,
    question_col=None,
):
    hf_pipe = load_hf_pipeline(task, model_name)
    if isinstance(hf_pipe, str):
        raise Exception(f"Failed to load model: {hf_pipe}")

    results      = []
    data_subset  = df.head(20)
    task_key     = task.lower().replace(" ", "-")

    for _, row in data_subset.iterrows():
        if task_key == "question-answering":
            res = hf_pipe(question=row[question_col], context=row[context_col])
            results.append(res['answer'])
        elif task_key == "summarization":
            res = hf_pipe(str(row[text_col]), max_length=130, min_length=30, do_sample=False)
            results.append(res[0]['summary_text'])
        elif task_key in ("sentiment-analysis", "text-classification"):
            res = hf_pipe(str(row[text_col]))
            label = res[0]['label']
            score = res[0].get('score', None)
            results.append(f"{label} ({score:.2f})" if score is not None else label)

    return results