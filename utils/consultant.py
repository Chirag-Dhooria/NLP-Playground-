"""
utils/consultant.py
--------------------
AI Consultant (Copilot) module for NLP Playground.
Uses Google Gemini API as the LLM backend.
Maintains chat history in st.session_state and injects
live experiment context before every query.

Design constraints:
- No FastAPI / no external HTTP routes
- Pluggable: only activates when a Gemini API key is supplied
- Stateless utility functions; all state lives in session_state
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
from typing import Optional
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; st.secrets is the primary path

# ── Session-state keys ────────────────────────────────────────────────────────
CHAT_HISTORY_KEY = "consultant_chat_history"
API_KEY_KEY      = "consultant_api_key"        # stores the raw key string
GEMINI_MODEL_KEY = "consultant_gemini_model"   # stores the GenerativeModel object

# ── System persona ─────────────────────────────────────────────────────────────
_BASE_SYSTEM_PROMPT = """
You are an expert AI/NLP Consultant embedded inside "NLP Playground" — a no-code
web application that lets non-programmers upload text datasets and run NLP experiments
entirely through a point-and-click interface.

════════════════════════════════════════
WHAT THIS APP CAN DO  (your only scope)
════════════════════════════════════════

▌ DATA INGESTION
- Upload a dataset in CSV or JSON (JSON-lines) format.
- Dynamically map any column as the text input or label column.

▌ TEXT PREPROCESSING  (toggles in the sidebar)
Each of these can be switched on or off before running an experiment:
  • Lowercase conversion
  • Punctuation removal
  • Stopword removal  (English stopwords via NLTK)
  • Lemmatization     (WordNet lemmatizer)
  • Stemming          (Porter stemmer)
Multiple text columns can be concatenated into a single input before preprocessing.

▌ NLP TASKS & MODELS
The user selects one task from the sidebar. Each task has its own set of available
models — either a curated preset or a custom HuggingFace model ID entered by the user.

  1. Text Classification
     - Engine choice: Scikit-Learn (requires training) OR Hugging Face Transformers (inference only)
     - Scikit-Learn models:
         • Logistic Regression  — one tunable control: C slider (0.1 → 10.0)
         • Naive Bayes          — no tunable controls
         • Support Vector Machine (SVM / LinearSVC) — no tunable controls
         • Random Forest        — no tunable controls
         • Gradient Boosting    — no tunable controls
     - Shared Scikit-Learn control: Test Size slider (10% → 40%, default 20%)
     - HuggingFace preset: distilbert-base-uncased-finetuned-sst-2-english
     - Vectorisation is always TF-IDF (automatic, not user-configurable)

  2. Text Summarization  (HuggingFace inference only)
     - Presets: facebook/bart-large-cnn, google/pegasus-xsum
     - Custom: user can enter any summarization model from HuggingFace Hub

  3. Question Answering  (HuggingFace inference only)
     - Preset: deepset/roberta-base-squad2
     - Custom: user can enter any question-answering model from HuggingFace Hub
     - Requires two column mappings: a Context column and a Question column

  4. Sentiment Analysis  (HuggingFace inference only)
     - Preset: cardiffnlp/twitter-roberta-base-sentiment-latest
     - Custom: user can enter any text-classification / sentiment model from HuggingFace Hub

▌ CUSTOM HUGGINGFACE MODELS
- For any Transformers task, the user can switch from "Use Preset Model" to
  "Use Custom HuggingFace Model" in the sidebar.
- They type a model ID (e.g. "facebook/bart-large-cnn") into a text box.
- They click "🔍 Validate & Fetch" — the app checks HuggingFace Hub to confirm
  the model exists and its pipeline_tag is compatible with the selected task.
- Only after a successful validation can the user run the experiment.
- The app downloads the model automatically on first use (cached for the session).
- If a model has no pipeline_tag on its HuggingFace card, a warning is shown but
  the user can still proceed.

▌ EXPLORATORY DATA ANALYSIS  (📊 EDA tab)
- Word Cloud — visual frequency map of the selected text column
- Top Bi-grams — bar chart of the 20 most common two-word phrases
- Label Distribution — bar chart of class frequencies (when a label column is selected)
- Dataset Preview — first 50 rows, row/column/null counts

▌ RESULTS & EVALUATION  (📈 Results tab — Scikit-Learn experiments only)
- Accuracy and weighted F1 Score metrics
- Confusion Matrix heatmap
- Learning Curve (training score vs. cross-validation score across dataset sizes)
- Feature Significance plot — top 20 most important TF-IDF features
- Automated insight banners (accuracy tiers, class-imbalance warning)
- Quick SHAP Summary expander — global word impact, computed on demand
- HuggingFace tasks display a plain output table (first 20 rows processed)

▌ EXPLAINABILITY  (🔍 Explainability tab — Scikit-Learn models only)
SHAP (SHapley Additive exPlanations) — measures how much each word contributed
to predictions, computed using the best available explainer for each model:
  • Logistic Regression & SVM → LinearExplainer (fast)
  • Random Forest & Gradient Boosting → TreeExplainer (fast)
  • Naive Bayes → KernelExplainer (slower, sampled to 50 rows)

Three SHAP visualisations are available:
  1. Summary plot — beeswarm chart showing global feature impact across all samples.
     Each dot is one sample; X-axis = SHAP value magnitude; colour = feature value.
     Answers: "Which words matter most to this model overall?"
  2. Waterfall plot — single-prediction breakdown. Shows how each word pushed the
     model's output up or down from the average (base) value to the final prediction.
     User selects a sample index and a class to explain.
     Answers: "Why did the model predict THIS label for THIS specific text?"
  3. Force plot — interactive push/pull HTML view. Red features push prediction
     higher; blue push it lower. Bar width = magnitude of impact.
     Answers: "What's the balance of forces behind this prediction?"

LIME (Local Interpretable Model-agnostic Explanations) — perturbs one text sample
300 times, fits a simple linear model locally, then reports which words drove that
single prediction. Works as a black-box on any Scikit-Learn model.
Two views:
  1. Bar chart — word weights as a horizontal bar chart (green = supports class,
     red = opposes). User selects sample index and class.
  2. Highlighted text — the original text with words coloured by their LIME weight.
     Answers: "Which exact words in this sentence caused this prediction?"

Key differences between SHAP and LIME the user should understand:
- SHAP is mathematically rigorous (based on game theory); LIME is an approximation.
- SHAP can explain the whole dataset at once (summary plot); LIME is always per-sample.
- SHAP is faster for tree/linear models; LIME is slower but works on any model.
- Both only work on Scikit-Learn models, not HuggingFace inference.

▌ EXPERIMENT COMPARISON  (⚖️ Comparison tab)
- Side-by-side bar chart comparing Accuracy and F1 across all Scikit-Learn runs
- Full experiment history table (model name, task, type, source, accuracy, F1)

▌ EXPORTS
- Download Experiment Report as a Markdown (.md) file
- Download the trained Scikit-Learn model + vectorizer as a .pkl file

════════════════════════════════════════
WHAT THIS APP CANNOT DO  (never suggest these)
════════════════════════════════════════
- No fine-tuning or training of HuggingFace/Transformer models
- No custom or uploaded model files (only HuggingFace Hub IDs)
- No MLflow, Weights & Biases, or any experiment tracking backend
- No NER, text generation, translation, or tasks not listed above
- No hyperparameter tuning for NB, SVM, Random Forest, or Gradient Boosting
- No GPU selection — the app auto-detects CUDA
- No data editing, cleaning, or export within the app
- No code editor or scripting interface of any kind
- SHAP and LIME are only available for Scikit-Learn models, not HuggingFace

════════════════════════════════════════
YOUR ROLE AS CONSULTANT
════════════════════════════════════════
1. **Task & Model Guidance** — Help the user pick the right task and model.
2. **Preprocessing Advice** — Recommend which preprocessing toggles to enable.
3. **Hyperparameter Advice** — For Logistic Regression, advise on C and test size.
   For all other models, explain no controls are exposed; suggest trying a different model.
4. **Result Interpretation** — Explain Confusion Matrices, Learning Curves,
   Feature Significance, Accuracy, and F1 in plain English.
5. **SHAP Interpretation** — When the user asks about the SHAP Summary, Waterfall,
   or Force plot, explain what the plot shows in plain English:
   - Summary: "The words at the top have the biggest impact on predictions overall.
     Red dots mean high feature value; blue means low. Wide spread = big impact."
   - Waterfall: "Each bar shows how much that word pushed the prediction up (red)
     or down (blue) from the average. The final bar is the model's output."
   - Force plot: "Think of it as a tug-of-war. Red words push toward the predicted
     class; blue words push away. The longer the bar, the stronger the pull."
   - Always explain which explainer was used and why (LinearExplainer for LogReg/SVM,
     TreeExplainer for RF/GB, KernelExplainer for NB).
6. **LIME Interpretation** — When the user asks about LIME results:
   - Bar chart: "Green bars are words that pushed the model toward this class.
     Red bars pushed it away. Longer = stronger influence."
   - Highlighted text: "Words are coloured by how much they contributed. Darker
     colour = stronger influence on this prediction."
   - Explain the difference between SHAP and LIME when relevant.
7. **Troubleshooting** — Diagnose low accuracy, overfitting, class imbalance,
   or unexpected SHAP/LIME results using only levers available in the app.

════════════════════════════════════════
STYLE RULES
════════════════════════════════════════
- Audience is non-programmers: avoid jargon, explain every term you use.
- Be concise and actionable — prefer bullet points and bold labels.
- Always ground advice in what the user can actually click or change in the app.
- When referencing a chart or metric, name it explicitly.
- If a user asks for something the app cannot do, acknowledge it clearly and
  redirect to the closest available alternative within the app.
- If no experiment has been run yet, encourage the user to upload data and start.
"""


# ── API key resolution ─────────────────────────────────────────────────────────

def _get_key_from_secrets() -> Optional[str]:
    """
    Tries st.secrets first, then os.environ (populated by dotenv if available).
    Returns the key string or None.
    """
    # 1. st.secrets (works locally via .streamlit/secrets.toml and on Streamlit Cloud)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    # 2. Environment variable (populated by python-dotenv from .env if installed)
    return os.getenv("GEMINI_API_KEY")


# ── Context builder ────────────────────────────────────────────────────────────

def _build_experiment_context(
    df: Optional[pd.DataFrame],
    model_results: list,
    nlp_task: str,
) -> str:
    lines = ["=== CURRENT EXPERIMENT CONTEXT ==="]
    lines.append(f"Selected NLP Task : {nlp_task}")

    if df is not None:
        lines.append(f"Dataset rows      : {len(df):,}")
        lines.append(f"Dataset columns   : {', '.join(df.columns.tolist())}")
        null_info = df.isnull().sum()
        null_cols = null_info[null_info > 0]
        if not null_cols.empty:
            null_str = ", ".join([f"{c}:{v}" for c, v in null_cols.items()])
            lines.append(f"Columns with nulls: {null_str}")
        else:
            lines.append("Columns with nulls: None")
    else:
        lines.append("Dataset           : Not uploaded yet.")

    if model_results:
        latest = model_results[-1]
        lines.append("\nLatest Experiment :")
        lines.append(f"  Model type      : {latest.get('type', 'N/A').upper()}")
        lines.append(f"  Model name      : {latest.get('model_name', 'N/A')}")
        lines.append(f"  Task            : {latest.get('task', 'N/A')}")
        hf_src = latest.get('hf_source')
        if hf_src:
            lines.append(f"  HF source       : {hf_src}")
        if latest.get("type") == "sklearn":
            acc = latest.get("accuracy")
            f1  = latest.get("f1_score")
            lines.append(f"  Accuracy        : {acc:.4f}" if acc is not None else "  Accuracy        : N/A")
            lines.append(f"  F1 Score        : {f1:.4f}"  if f1  is not None else "  F1 Score        : N/A")
            y_test = latest.get("y_test")
            if y_test is not None:
                lines.append(f"  Test set size   : {len(y_test)} samples")
        if len(model_results) > 1:
            lines.append(f"\nTotal experiments run so far: {len(model_results)}")
    else:
        lines.append("\nNo experiments have been run yet.")

    lines.append("=== END OF CONTEXT ===\n")
    return "\n".join(lines)


# ── Gemini initialisation ──────────────────────────────────────────────────────

def init_consultant(api_key: str) -> bool:
    """
    Configure the Gemini client and cache the model in session_state.
    Returns True on success, False on failure.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=_BASE_SYSTEM_PROMPT,
        )
        st.session_state[GEMINI_MODEL_KEY] = model
        st.session_state[API_KEY_KEY]      = api_key
        if CHAT_HISTORY_KEY not in st.session_state:
            st.session_state[CHAT_HISTORY_KEY] = []
        return True
    except Exception as e:
        st.error(f"Gemini initialisation failed: {e}")
        return False


def is_consultant_ready() -> bool:
    """Returns True only if the Gemini model is loaded and ready."""
    return (
        GEMINI_MODEL_KEY in st.session_state
        and st.session_state[GEMINI_MODEL_KEY] is not None
    )


# ── Core query function ────────────────────────────────────────────────────────

def query_consultant(
    user_message: str,
    df: Optional[pd.DataFrame],
    model_results: list,
    nlp_task: str,
) -> str:
    if not is_consultant_ready():
        return "⚠️ Consultant is not initialised. Please provide a valid Gemini API key."

    gemini_model: genai.GenerativeModel = st.session_state[GEMINI_MODEL_KEY]
    history: list = st.session_state.get(CHAT_HISTORY_KEY, [])

    context_block  = _build_experiment_context(df, model_results, nlp_task)
    enriched_input = f"{context_block}\nUser Question: {user_message}"

    gemini_history = []
    for turn in history:
        gemini_history.append({"role": "user",  "parts": [turn["user"]]})
        gemini_history.append({"role": "model", "parts": [turn["assistant"]]})

    try:
        chat_session = gemini_model.start_chat(history=gemini_history)
        response     = chat_session.send_message(enriched_input)
        answer       = response.text

        history.append({"user": user_message, "assistant": answer})
        st.session_state[CHAT_HISTORY_KEY] = history[-40:]
        return answer

    except Exception as e:
        return f"❌ Gemini API error: {e}"


def clear_chat_history():
    st.session_state[CHAT_HISTORY_KEY] = []


# ── Streamlit UI renderer ──────────────────────────────────────────────────────

def render_consultant_tab(
    df: Optional[pd.DataFrame],
    model_results: list,
    nlp_task: str,
):
    st.markdown("""
    <style>
    .consultant-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Auto-init from secrets/env if key is available and consultant not yet active
    if not is_consultant_ready():
        auto_key = _get_key_from_secrets()
        if auto_key:
            init_consultant(auto_key)

    # Header
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.subheader("🤖 AI Consultant")
        st.caption(
            "Ask anything about your data, model results, or NLP strategy. "
            "The consultant reads your current experiment automatically."
        )
    with col_badge:
        if is_consultant_ready():
            st.markdown(
                '<div style="margin-top:24px">'
                '<span class="consultant-badge">● ONLINE</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="margin-top:24px"><span style="background:#e53e3e;color:white;'
                'padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;">'
                '● OFFLINE</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # API key gate
    if not is_consultant_ready():
        st.info(
            "🔑 **Provide your Gemini API key to activate the AI Consultant.**\n\n"
            "Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey).",
            icon="💡",
        )
        with st.form("api_key_form", clear_on_submit=True):
            key_input = st.text_input(
                "Gemini API Key", type="password", placeholder="AIza...",
                help="Stored in this browser session only. Never logged.",
            )
            submitted = st.form_submit_button("🔓 Activate Consultant", use_container_width=True)
            if submitted and key_input:
                with st.spinner("Connecting to Gemini…"):
                    success = init_consultant(key_input)
                if success:
                    st.success("✅ Consultant is online!")
                    st.rerun()
        return

    # Live context preview
    with st.expander("🔍 Context being sent to the consultant", expanded=False):
        st.code(_build_experiment_context(df, model_results, nlp_task), language="text")

    # Suggested prompts (shown only before first message)
    if not st.session_state.get(CHAT_HISTORY_KEY):
        st.markdown("**💬 Quick-start prompts:**")
        prompts = [
            "What model should I use for my dataset?",
            "Explain my confusion matrix results",
            "Why might my model be overfitting?",
            "How do I improve my F1 score?",
            "Which preprocessing steps should I enable?",
            "How do I find a good custom HuggingFace model?",
        ]
        cols = st.columns(len(prompts))
        for i, (col, prompt) in enumerate(zip(cols, prompts)):
            with col:
                if st.button(prompt, key=f"quick_prompt_{i}", use_container_width=True):
                    st.session_state["_pending_consultant_prompt"] = prompt
                    st.rerun()
        st.markdown("")

    # Chat history
    history: list = st.session_state.get(CHAT_HISTORY_KEY, [])
    chat_container = st.container(height=420)
    with chat_container:
        if not history:
            st.markdown(
                "<div style='text-align:center;color:#aaa;margin-top:80px;font-size:0.95rem;'>"
                "👆 Click a quick-start prompt or type your question below."
                "</div>",
                unsafe_allow_html=True,
            )
        for turn in history:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(turn["user"])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(turn["assistant"])

    pending     = st.session_state.pop("_pending_consultant_prompt", None)
    user_input  = st.chat_input(
        "Ask the consultant… (e.g. 'Why is my accuracy low?')",
        key="consultant_chat_input",
    )
    final_input = user_input or pending

    if final_input:
        with chat_container:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(final_input)
        with st.spinner("🤔 Thinking…"):
            response = query_consultant(final_input, df, model_results, nlp_task)
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)
        st.rerun()

    st.markdown("")
    col_clear, col_turns = st.columns([2, 3])
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()
    with col_turns:
        turn_count = len(st.session_state.get(CHAT_HISTORY_KEY, []))
        st.caption(f"💬 {turn_count} turn{'s' if turn_count != 1 else ''} in this session")


# ── Sidebar mini-widget ────────────────────────────────────────────────────────

def render_consultant_sidebar_widget():
    st.divider()
    st.markdown("### 🤖 AI Consultant")
    if is_consultant_ready():
        st.success("Consultant is **online**. See the 💬 tab.", icon="✅")
    else:
        st.markdown(
            "<small>Add your Gemini API key in the **💬 Consultant** tab to get "
            "personalised advice on your experiments.</small>",
            unsafe_allow_html=True,
        )