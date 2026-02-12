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
from utils.model_handler import get_model, train_model, evaluate_model

st.set_page_config(page_title="NLP Playground", page_icon="🛝", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = []

def generate_automated_report(result, df):
    report = f"# NLP Playground: Automated Experiment Report\n\n"
    report += f"## 1. Dataset Summary\n"
    report += f"- Total Rows: {len(df)}\n"
    report += f"- Columns Detected: {', '.join(df.columns.tolist())}\n\n"
    report += f"## 2. Preprocessing Configuration\n"
    report += f"- Operations Applied: {result['preprocessing']}\n\n"
    report += f"## 3. Model & Training Details\n"
    report += f"- Algorithm: {result['model_name']}\n"
    report += f"- Hyperparameters: {result['hyperparameters']}\n"
    report += f"- Test Split Ratio: {result['test_size']}\n\n"
    report += f"## 4. Performance Insights\n"
    report += f"- Accuracy Score: {result['accuracy']:.4f}\n"
    report += f"- Weighted F1-Score: {result['f1_score']:.4f}\n\n"
    report += f"## 5. Visual Insights Summary\n"
    report += f"- The model was trained using TF-IDF feature extraction.\n"
    report += f"- Learning curves and confusion matrices are available in the dashboard for detailed error analysis.\n"
    return report

st.title("NLP Playground 🛝")
st.markdown("""
Welcome to the NLP Playground! This is a no-code platform designed to help you explore, preprocess, and analyze your text data. 
You can train classical machine learning models for text classification and compare their performance, all without writing a single line of code.
""")

with st.sidebar:
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.success("File uploaded successfully!")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.header("2. Column Selection")
        text_column = st.selectbox("Select Text Column", df.columns)
        target_column = st.selectbox("Select Target/Label Column", df.columns)

        st.header("3. Preprocessing")
        options = {
            'lowercase': st.checkbox("Convert to Lowercase", value=True),
            'remove_punctuation': st.checkbox("Remove Punctuation", value=True),
            'remove_stopwords': st.checkbox("Remove Stopwords"),
            'lemmatization': st.checkbox("Lemmatization"),
            'stemming': st.checkbox("Stemming")
        }

        st.header("4. Model Selection")
        model_name = st.selectbox(
            "Choose a Model",
            ["Logistic Regression", "Naive Bayes", "Support Vector Machine (SVM)", "Random Forest", "Gradient Boosting"]
        )
        
        st.subheader("Model Hyperparameters")
        params = {}
        if model_name == "Logistic Regression":
            params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
            params['max_iter'] = st.select_slider("Max Iterations", options=[100, 500, 1000], value=100)
        elif model_name == "Naive Bayes":
            params['alpha'] = st.slider("Alpha", 0.0, 2.0, 1.0)
        elif model_name == "Support Vector Machine (SVM)":
            params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
        elif model_name == "Random Forest":
            params['n_estimators'] = st.slider("Trees", 50, 500, 100, 50)
            params['max_depth'] = st.slider("Max Depth", 5, 50, 10)
        elif model_name == "Gradient Boosting":
            params['n_estimators'] = st.slider("Estimators", 50, 500, 100, 50)
            params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1)

        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

        if st.button("🚀 Run Experiment", use_container_width=True):
            with st.status("Running experiment...", expanded=True) as status:
                st.write("Step 1: Preprocessing...")
                processed_df = preprocess_text(df.copy(), text_column, options)
                
                status.update(label="Step 2: Training...")
                model = get_model(model_name, params=params)
                trained_model, vec, X_tr, y_tr, X_te, y_te = train_model(
                    processed_df, 'processed_text', target_column, model, test_size
                )
                
                status.update(label="Step 3: Evaluating...")
                metrics, y_pred = evaluate_model(trained_model, X_te, y_te)
                
                run_result = {
                    'model_name': model_name,
                    'hyperparameters': str(params),
                    'test_size': test_size,
                    'preprocessing': ", ".join([k for k, v in options.items() if v]),
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'trained_model': trained_model,
                    'vectorizer': vec,
                    'X_train_vec': X_tr,
                    'y_train': y_tr,
                    'y_test': y_te,
                    'y_pred': y_pred
                }
                st.session_state.model_results.append(run_result)
                status.update(label="Complete!", state="complete", expanded=False)
            st.success("Experiment finished!")

if st.session_state.df is None:
    st.info("Upload data to begin.")
else:
    tabs = st.tabs(["Data & EDA", "Latest Model Results", "Model Comparison"])

    with tabs[0]:
        st.header("Data Preview")
        st.dataframe(st.session_state.df.head())
        st.header("Exploratory Data Analysis")
        cols = st.columns(2)
        cols[0].pyplot(plot_label_distribution(st.session_state.df, target_column))
        cols[1].pyplot(generate_wordcloud(st.session_state.df, text_column))
        st.pyplot(plot_top_ngrams(st.session_state.df, text_column))

    with tabs[1]:
        if st.session_state.model_results:
            latest = st.session_state.model_results[-1]
            st.header(f"Results: {latest['model_name']}")
            m_cols = st.columns(2)
            m_cols[0].metric("Accuracy", f"{latest['accuracy']:.4f}")
            m_cols[1].metric("F1-Score", f"{latest['f1_score']:.4f}")

            st.subheader("Performance Visualizations")
            v_cols = st.columns(2)
            v_cols[0].pyplot(plot_confusion_matrix(latest['y_test'], latest['y_pred']))
            v_cols[1].pyplot(plot_learning_curve(get_model(latest['model_name']), latest['X_train_vec'], latest['y_train']))

            st.subheader("Feature Significance")
            fig_imp = plot_feature_importance(latest['trained_model'], latest['vectorizer'])
            if fig_imp: st.pyplot(fig_imp)

            st.header("Automated Insights & Export")
            exp_cols = st.columns(3)
            report_text = generate_automated_report(latest, st.session_state.df)
            exp_cols[0].download_button("Download Full Report (.md)", report_text, f"report_{latest['model_name']}.md")
            pkl_data = pickle.dumps(latest['trained_model'])
            exp_cols[1].download_button("Download Model (.pkl)", pkl_data, f"{latest['model_name']}.pkl")
            csv_rep = pd.DataFrame({'Metric':['Accuracy','F1'], 'Score':[latest['accuracy'], latest['f1_score']]}).to_csv(index=False).encode('utf-8')
            exp_cols[2].download_button("Download Metrics (.csv)", csv_rep, f"metrics_{latest['model_name']}.csv")

    with tabs[2]:
        if st.session_state.model_results:
            st.header("Performance Benchmarking")
            st.pyplot(plot_model_comparison(st.session_state.model_results))
            st.dataframe(pd.DataFrame([{
                "Model": r['model_name'], "Accuracy": f"{r['accuracy']:.4f}", 
                "F1": f"{r['f1_score']:.4f}", "Preprocessing": r['preprocessing']
            } for r in st.session_state.model_results]))
            if st.button("Clear Results"):
                st.session_state.model_results = []; st.rerun()