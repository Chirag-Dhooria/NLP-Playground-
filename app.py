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
from utils.model_handler import run_sklearn_pipeline, run_hf_inference, get_sklearn_model

st.set_page_config(page_title="NLP Playground", page_icon="🛝", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = []

def generate_automated_report(result, df):
    report = f"# NLP Playground: Experiment Report\n\n"
    report += f"## Dataset Summary\n- Total Rows: {len(df)}\n\n"
    report += f"## Model Details\n- Task: {result.get('task')}\n- Model: {result['model_name']}\n"
    if 'accuracy' in result:
        report += f"\n## Performance\n- Accuracy: {result['accuracy']:.4f}\n- F1: {result['f1_score']:.4f}\n"
    return report

st.title("NLP Playground 🛝")
st.markdown("A no-code platform to explore text data and evaluate NLP models.")

with st.sidebar:
    st.header("1. Task Selection")
    nlp_task = st.selectbox("Select NLP Task", ["Text Classification", "Text Summarization", "Question Answering", "Sentiment Analysis"])

    st.header("2. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.success("Data Loaded")

    if st.session_state.df is not None:
        st.header("3. Mapping & Preprocessing")
        if nlp_task == "Question Answering":
            ctx = st.selectbox("Context", st.session_state.df.columns)
            qst = st.selectbox("Question", st.session_state.df.columns)
            text_cols = [ctx, qst]
        else:
            text_cols = st.multiselect("Input Columns", st.session_state.df.columns, default=[st.session_state.df.columns[0]])
            target_col = st.selectbox("Label Column", [None] + list(st.session_state.df.columns))

        options = {'lowercase': st.checkbox("Lowercase", True), 'remove_punctuation': st.checkbox("Punctuation", True), 'remove_stopwords': st.checkbox("Stopwords"), 'lemmatization': st.checkbox("Lemmatize"), 'stemming': st.checkbox("Stemming")}

        st.header("4. Model Selection")
        if nlp_task == "Text Classification":
            source = st.radio("Source", ["Scikit-Learn", "Transformers"])
        else:
            source = "Transformers"

        if source == "Scikit-Learn":
            m_name = st.selectbox("Model", ["Logistic Regression", "Naive Bayes", "Support Vector Machine (SVM)", "Random Forest", "Gradient Boosting"])
            params = {'C': st.slider("C", 0.1, 10.0, 1.0)} if "Logistic" in m_name else {}
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        else:
            presets = {"Text Classification": ["distilbert-base-uncased-finetuned-sst-2-english"], "Text Summarization": ["facebook/bart-large-cnn", "google/pegasus-xsum"], "Question Answering": ["deepset/roberta-base-squad2"], "Sentiment Analysis": ["cardiffnlp/twitter-roberta-base-sentiment-latest"]}
            m_name = st.selectbox("Hugging Face Presets", presets[nlp_task])

        if st.button("🚀 Run Experiment", use_container_width=True):
            with st.status("Executing...") as status:
                is_qa = nlp_task == "Question Answering"
                proc_df = preprocess_text(st.session_state.df.copy(), text_cols, options, is_qa, ctx if is_qa else None, qst if is_qa else None)
                if source == "Scikit-Learn":
                    model_obj, vec, met, X_tr, y_tr, y_te, y_pred = run_sklearn_pipeline(proc_df, 'processed_text', target_col, m_name, params, test_size)
                    st.session_state.model_results.append({'type': 'sklearn', 'task': nlp_task, 'model_name': m_name, 'accuracy': met['accuracy'], 'f1_score': met['f1_score'], 'y_test': y_te, 'y_pred': y_pred, 'trained_model': model_obj, 'vectorizer': vec, 'X_train_vec': X_tr, 'y_train': y_tr})
                else:
                    results = run_hf_inference(proc_df, nlp_task.lower().replace(" ", "-"), m_name, 'processed_text' if not is_qa else None, ctx if is_qa else None, qst if is_qa else None)
                    st.session_state.model_results.append({'type': 'hf', 'task': nlp_task, 'model_name': m_name, 'results': results})
                status.update(label="Complete!", state="complete")

if st.session_state.df is not None:
    t_eda, t_res, t_comp = st.tabs(["📊 EDA", "📈 Results", "⚖️ Comparison"])
    with t_eda:
        st.subheader("Data Insights")
        c1, c2 = st.columns(2)
        if nlp_task != "Question Answering":
            c1.pyplot(generate_wordcloud(st.session_state.df, text_cols[0]))
            c2.pyplot(plot_top_ngrams(st.session_state.df, text_cols[0]))
    with t_res:
        if st.session_state.model_results:
            latest = st.session_state.model_results[-1]
            if latest['type'] == 'sklearn':
                st.metric("Accuracy", f"{latest['accuracy']:.4f}")
                v1, v2 = st.columns(2)
                v1.pyplot(plot_confusion_matrix(latest['y_test'], latest['y_pred']))
                v2.pyplot(plot_learning_curve(get_sklearn_model(latest['model_name']), latest['X_train_vec'], latest['y_train']))
                st.pyplot(plot_feature_importance(latest['trained_model'], latest['vectorizer']))
            else:
                st.table(pd.DataFrame(latest['results'], columns=["Outputs"]).head(10))
            st.download_button("Download Report", generate_automated_report(latest, st.session_state.df), "report.md")
    with t_comp:
        if st.session_state.model_results:
            sk_runs = [r for r in st.session_state.model_results if r['type'] == 'sklearn']
            if sk_runs: st.pyplot(plot_model_comparison(sk_runs))
            st.dataframe(pd.DataFrame([{"Model": r['model_name'], "Accuracy": r.get('accuracy', 'N/A')} for r in st.session_state.model_results]))