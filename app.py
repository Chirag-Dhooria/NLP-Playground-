import streamlit as st
import pandas as pd
import pickle
from utils.data_loader import load_data
from utils.preprocessing import preprocess_text
from utils.visualizer import (
    plot_label_distribution, generate_wordcloud,
    plot_confusion_matrix, plot_top_ngrams,
    plot_model_comparison, plot_feature_importance,
    plot_learning_curve
)
from utils.model_handler import (
    run_sklearn_pipeline, run_hf_inference,
    get_sklearn_model, validate_hf_model
)
from utils.consultant import render_consultant_tab, render_consultant_sidebar_widget

st.set_page_config(page_title="NLP Playground", page_icon="🛝", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #a0aec0;
    margin-bottom: 4px;
}
[data-testid="stTabs"] button {
    font-size: 0.9rem;
    font-weight: 600;
}
[data-testid="stMetric"] {
    background: #f7fafc;
    border-left: 4px solid #667eea;
    padding: 10px 16px;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Session-state initialisation ───────────────────────────────────────────────
if 'df'            not in st.session_state: st.session_state.df            = None
if 'model_results' not in st.session_state: st.session_state.model_results = []
# Stores validated custom model: {"model_id": str, "task": str, "pipeline_tag": str|None} | None
if 'validated_custom_model' not in st.session_state:
    st.session_state.validated_custom_model = None


# ── Automated report generator ─────────────────────────────────────────────────
def generate_automated_report(result: dict, df: pd.DataFrame) -> str:
    report  = "# NLP Playground: Experiment Report\n\n"
    report += f"## Dataset Summary\n- Total Rows: {len(df)}\n\n"
    report += f"## Model Details\n- Task: {result.get('task')}\n- Model: {result['model_name']}\n"
    if result.get('hf_source') == 'custom':
        report += "- Model Source: Custom (user-provided HuggingFace ID)\n"
    if 'accuracy' in result:
        report += f"\n## Performance\n- Accuracy: {result['accuracy']:.4f}\n- F1: {result['f1_score']:.4f}\n"
    report += "\n## Automated Insights\n"
    if result.get('type') == 'sklearn':
        acc = result.get('accuracy', 0)
        f1  = result.get('f1_score', 0)
        if acc >= 0.90:
            report += "- ✅ **Excellent accuracy** (≥ 90%). Model generalises well.\n"
        elif acc >= 0.75:
            report += "- 🟡 **Good accuracy** (75–90%). Consider tuning hyperparameters.\n"
        else:
            report += "- 🔴 **Low accuracy** (< 75%). Try more preprocessing or a stronger model.\n"
        if abs(acc - f1) > 0.05:
            report += "- ⚠️ Accuracy and F1 differ significantly — possible class imbalance.\n"
    return report


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Task Selection")
    nlp_task = st.selectbox(
        "Select NLP Task",
        ["Text Classification", "Text Summarization", "Question Answering", "Sentiment Analysis"]
    )

    st.header("2. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.success("✅ Data Loaded")

    if st.session_state.df is not None:
        st.header("3. Mapping & Preprocessing")
        if nlp_task == "Question Answering":
            ctx       = st.selectbox("Context Column",  st.session_state.df.columns)
            qst       = st.selectbox("Question Column", st.session_state.df.columns)
            text_cols = [ctx, qst]
        else:
            text_cols  = st.multiselect(
                "Input Columns",
                st.session_state.df.columns,
                default=[st.session_state.df.columns[0]]
            )
            target_col = st.selectbox("Label Column", [None] + list(st.session_state.df.columns))

        options = {
            'lowercase':          st.checkbox("Lowercase",   True),
            'remove_punctuation': st.checkbox("Punctuation", True),
            'remove_stopwords':   st.checkbox("Stopwords"),
            'lemmatization':      st.checkbox("Lemmatize"),
            'stemming':           st.checkbox("Stemming"),
        }

        st.header("4. Model Selection")
        source = st.radio("Source", ["Scikit-Learn", "Transformers"]) \
            if nlp_task == "Text Classification" else "Transformers"

        # ── Scikit-Learn branch ───────────────────────────────────────────────
        if source == "Scikit-Learn":
            m_name = st.selectbox(
                "Model",
                ["Logistic Regression", "Naive Bayes", "Support Vector Machine (SVM)",
                 "Random Forest", "Gradient Boosting"]
            )
            params    = {'C': st.slider("C (Regularisation)", 0.1, 10.0, 1.0)} \
                        if "Logistic" in m_name else {}
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
            hf_source_label = "sklearn"

        # ── Transformers branch ───────────────────────────────────────────────
        else:
            presets = {
                "Text Classification": ["distilbert-base-uncased-finetuned-sst-2-english"],
                "Text Summarization":  ["facebook/bart-large-cnn", "google/pegasus-xsum"],
                "Question Answering":  ["deepset/roberta-base-squad2"],
                "Sentiment Analysis":  ["cardiffnlp/twitter-roberta-base-sentiment-latest"],
            }

            model_source_mode = st.radio(
                "Model Source",
                ["Use Preset Model", "Use Custom HuggingFace Model"],
            )

            # ── Preset path ───────────────────────────────────────────────────
            if model_source_mode == "Use Preset Model":
                m_name = st.selectbox("HuggingFace Presets", presets[nlp_task])
                st.session_state.validated_custom_model = None
                hf_source_label = "preset"

            # ── Custom model path ─────────────────────────────────────────────
            else:
                st.caption(
                    "Enter any public model ID from "
                    "[huggingface.co/models](https://huggingface.co/models). "
                    "The app will verify it's compatible before running."
                )

                custom_input = st.text_input(
                    "HuggingFace Model ID",
                    placeholder="e.g. facebook/bart-large-cnn",
                    key="custom_hf_model_input",
                )

                # Show validated state persistently
                vmc = st.session_state.validated_custom_model
                if vmc and vmc.get("task") == nlp_task:
                    st.success(f"✅ Ready: `{vmc['model_id']}`")

                col_v, col_c = st.columns([3, 1])
                with col_v:
                    validate_clicked = st.button(
                        "🔍 Validate & Fetch",
                        use_container_width=True,
                        key="validate_hf_btn",
                    )
                with col_c:
                    if st.button("✖", key="clear_custom_btn", help="Clear"):
                        st.session_state.validated_custom_model = None
                        st.rerun()

                if validate_clicked:
                    if not custom_input.strip():
                        st.warning("Please enter a model ID first.")
                    else:
                        with st.spinner(f"Checking `{custom_input.strip()}` on HuggingFace…"):
                            val_result = validate_hf_model(custom_input.strip(), nlp_task)
                        if val_result["valid"]:
                            st.session_state.validated_custom_model = {
                                "model_id":     val_result["model_id"],
                                "task":         nlp_task,
                                "pipeline_tag": val_result["pipeline_tag"],
                            }
                            st.success(val_result["message"])
                        else:
                            st.session_state.validated_custom_model = None
                            st.error(val_result["message"])

                # Resolve which model name to run
                vmc = st.session_state.validated_custom_model
                if vmc and vmc.get("task") == nlp_task:
                    m_name          = vmc["model_id"]
                    hf_source_label = "custom"
                else:
                    m_name          = presets[nlp_task][0]
                    hf_source_label = "custom_unvalidated"

        # ── Run button ────────────────────────────────────────────────────────
        st.divider()
        run_clicked = st.button("🚀 Run Experiment", use_container_width=True, type="primary")

        if run_clicked:
            # Guard: block unvalidated custom models
            if (source == "Transformers"
                    and model_source_mode == "Use Custom HuggingFace Model"
                    and hf_source_label == "custom_unvalidated"):
                st.error(
                    "⚠️ Validate your custom model first — click **🔍 Validate & Fetch** above."
                )
            else:
                with st.status("Executing experiment…") as status:
                    try:
                        is_qa   = nlp_task == "Question Answering"
                        proc_df = preprocess_text(
                            st.session_state.df.copy(), text_cols, options,
                            is_qa,
                            ctx if is_qa else None,
                            qst if is_qa else None,
                        )

                        if source == "Scikit-Learn":
                            model_obj, vec, met, X_tr, y_tr, y_te, y_pred = run_sklearn_pipeline(
                                proc_df, 'processed_text', target_col, m_name, params, test_size
                            )
                            st.session_state.model_results.append({
                                'type':          'sklearn',
                                'task':          nlp_task,
                                'model_name':    m_name,
                                'accuracy':      met['accuracy'],
                                'f1_score':      met['f1_score'],
                                'y_test':        y_te,
                                'y_pred':        y_pred,
                                'trained_model': model_obj,
                                'vectorizer':    vec,
                                'X_train_vec':   X_tr,
                                'y_train':       y_tr,
                            })
                        else:
                            status.update(label=f"⬇️ Loading `{m_name}` from HuggingFace…")
                            results = run_hf_inference(
                                proc_df,
                                nlp_task.lower().replace(" ", "-"),
                                m_name,
                                'processed_text' if not is_qa else None,
                                ctx if is_qa else None,
                                qst if is_qa else None,
                            )
                            st.session_state.model_results.append({
                                'type':       'hf',
                                'task':       nlp_task,
                                'model_name': m_name,
                                'hf_source':  hf_source_label,
                                'results':    results,
                            })
                        status.update(label="✅ Complete!", state="complete")

                    except Exception as e:
                        status.update(label="❌ Experiment failed", state="error")
                        st.error(f"Error: {e}")

    render_consultant_sidebar_widget()


# ── Main panel ─────────────────────────────────────────────────────────────────
st.title("NLP Playground 🛝")
st.markdown("A no-code platform to explore text data and evaluate NLP models.")

if st.session_state.df is not None:
    t_eda, t_res, t_comp, t_consultant = st.tabs([
        "📊 EDA", "📈 Results", "⚖️ Comparison", "💬 AI Consultant"
    ])

    # ── EDA tab ───────────────────────────────────────────────────────────────
    with t_eda:
        st.subheader("Exploratory Data Analysis")
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(st.session_state.df.head(50), use_container_width=True)
            c_rows, c_cols, c_nulls = st.columns(3)
            c_rows.metric("Rows",         f"{len(st.session_state.df):,}")
            c_cols.metric("Columns",      len(st.session_state.df.columns))
            c_nulls.metric("Total Nulls", int(st.session_state.df.isnull().sum().sum()))
        st.divider()
        if nlp_task != "Question Answering" and text_cols:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Word Cloud**")
                st.pyplot(generate_wordcloud(st.session_state.df, text_cols[0]))
            with c2:
                st.markdown("**Top Bi-grams**")
                st.pyplot(plot_top_ngrams(st.session_state.df, text_cols[0]))
            if 'target_col' in dir() and target_col:
                st.markdown("**Label Distribution**")
                st.pyplot(plot_label_distribution(st.session_state.df, target_col))

    # ── Results tab ───────────────────────────────────────────────────────────
    with t_res:
        if not st.session_state.model_results:
            st.info("No experiments run yet. Configure your model in the sidebar and click **🚀 Run Experiment**.")
        else:
            latest = st.session_state.model_results[-1]
            if latest.get('hf_source') == 'custom':
                st.caption("🤗 Results from a custom HuggingFace model")
            st.subheader(f"Latest: {latest['model_name']}")

            if latest['type'] == 'sklearn':
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy",            f"{latest['accuracy']:.4f}")
                m2.metric("F1 Score (weighted)", f"{latest['f1_score']:.4f}")
                m3.metric("Task",                latest['task'])
                acc = latest['accuracy']
                if acc >= 0.90:
                    st.success("✅ **Excellent accuracy** (≥ 90%). Your model generalises very well.")
                elif acc >= 0.75:
                    st.warning("🟡 **Good accuracy** (75–90%). Consider tuning or adding more data.")
                else:
                    st.error("🔴 **Low accuracy** (< 75%). Try different preprocessing or a different model.")
                if abs(latest['accuracy'] - latest['f1_score']) > 0.05:
                    st.warning("⚠️ Accuracy and F1 differ significantly — possible class imbalance.")
                st.divider()
                v1, v2 = st.columns(2)
                with v1:
                    st.markdown("**Confusion Matrix**")
                    st.pyplot(plot_confusion_matrix(latest['y_test'], latest['y_pred']))
                with v2:
                    st.markdown("**Learning Curve**")
                    st.pyplot(plot_learning_curve(
                        get_sklearn_model(latest['model_name']),
                        latest['X_train_vec'], latest['y_train'],
                    ))
                st.markdown("**Feature Significance (Top 20 TF-IDF Features)**")
                fi_fig = plot_feature_importance(latest['trained_model'], latest['vectorizer'])
                if fi_fig:
                    st.pyplot(fi_fig)
                else:
                    st.info("Feature importance is not available for this model type.")
                st.divider()
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "⬇️ Download Report (.md)",
                        generate_automated_report(latest, st.session_state.df),
                        file_name="nlp_playground_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col_dl2:
                    model_bytes = pickle.dumps({
                        'model': latest['trained_model'], 'vectorizer': latest['vectorizer'],
                    })
                    st.download_button(
                        "⬇️ Download Model (.pkl)",
                        data=model_bytes,
                        file_name=f"{latest['model_name'].replace(' ', '_')}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
            else:
                st.markdown(f"**Model:** `{latest['model_name']}`")
                st.markdown(f"**Task:** {latest['task']}")
                if latest.get('hf_source') == 'custom':
                    st.info("ℹ️ Results generated using a custom HuggingFace model (first 20 rows).")
                st.dataframe(
                    pd.DataFrame(latest['results'], columns=["Output"]),
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Download Report (.md)",
                    generate_automated_report(latest, st.session_state.df),
                    file_name="nlp_playground_report.md",
                    mime="text/markdown",
                )

    # ── Comparison tab ────────────────────────────────────────────────────────
    with t_comp:
        if not st.session_state.model_results:
            st.info("Run at least one experiment to see comparison data here.")
        else:
            st.subheader("Experiment Comparison Dashboard")
            sk_runs = [r for r in st.session_state.model_results if r['type'] == 'sklearn']
            if sk_runs:
                st.markdown("**Performance Chart (Sklearn Models)**")
                st.pyplot(plot_model_comparison(sk_runs))
                st.divider()
            st.markdown("**All Experiments**")
            all_rows = []
            for r in st.session_state.model_results:
                all_rows.append({
                    "Model":    r['model_name'],
                    "Task":     r.get('task', 'N/A'),
                    "Type":     r.get('type', 'N/A').upper(),
                    "Source":   r.get('hf_source', '—').title() if r.get('type') == 'hf' else "Scikit-Learn",
                    "Accuracy": f"{r['accuracy']:.4f}" if 'accuracy' in r else "—",
                    "F1 Score": f"{r['f1_score']:.4f}" if 'f1_score' in r else "—",
                })
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True)
            if st.button("🗑️ Clear All Experiments", use_container_width=False):
                st.session_state.model_results = []
                st.rerun()

    # ── AI Consultant tab ─────────────────────────────────────────────────────
    with t_consultant:
        render_consultant_tab(
            df=st.session_state.df,
            model_results=st.session_state.model_results,
            nlp_task=nlp_task,
        )

else:
    st.markdown("---")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
        ### 👋 Welcome to NLP Playground!
        Get started in 4 simple steps:
        1. **Select an NLP Task** in the sidebar
        2. **Upload a CSV or JSON** dataset
        3. **Map your columns** and choose preprocessing options
        4. **Pick a model** and click 🚀 Run Experiment

        Need help? Activate the **💬 AI Consultant** tab after uploading your data.
        """)
    with col_r:
        st.markdown("""
        #### 🧩 Supported Tasks
        | Task | Engine |
        |---|---|
        | Text Classification | Scikit-Learn / 🤗 |
        | Text Summarization | 🤗 Transformers |
        | Question Answering | 🤗 Transformers |
        | Sentiment Analysis | 🤗 Transformers |
        """)