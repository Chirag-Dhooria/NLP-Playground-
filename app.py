import streamlit as st
import pandas as pd
import pickle
import base64
from utils.data_loader import load_data
from utils.preprocessing import preprocess_text
from utils.visualizer import plot_label_distribution, generate_wordcloud, plot_confusion_matrix, plot_top_ngrams
from utils.model_handler import get_model, train_model, evaluate_model

# --- Page Config ---
st.set_page_config(page_title="NLP Playground", page_icon="🚀", layout="wide")

# --- State Management ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = []

# --- Header ---
st.title("NLP Playground 🚀")
st.markdown("A no-code platform to preprocess data, train, and compare NLP models.")

# --- Sidebar UI ---
with st.sidebar:
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.success("File uploaded successfully!")
            st.session_state.model_results = [] # Clear previous results on new file

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
        
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

        if st.button("🚀 Run Experiment", use_container_width=True):
            with st.spinner("Running experiment..."):
                processed_df = preprocess_text(df.copy(), text_column, options)
                model = get_model(model_name)
                
                trained_model, vectorizer, X_test_vec, y_test = train_model(
                    processed_df, 'processed_text', target_column, model, test_size
                )
                
                metrics, y_pred = evaluate_model(trained_model, X_test_vec, y_test)
                
                # Store results for display and comparison
                run_result = {
                    'model_name': model_name,
                    'test_size': test_size,
                    'preprocessing': ", ".join([k for k, v in options.items() if v]),
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'trained_model': trained_model,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                st.session_state.model_results.append(run_result)
            st.success("Experiment finished!")

# --- Main Panel ---
if st.session_state.df is None:
    st.info("Please upload a dataset to begin.")
else:
    tabs = st.tabs(["Data & EDA", "Latest Model Results", "Model Comparison"])

    with tabs[0]:
        st.header("Data Preview")
        st.dataframe(st.session_state.df.head())

        st.header("Exploratory Data Analysis")
        eda_cols = st.columns(2)
        with eda_cols[0]:
            st.subheader("Label Distribution")
            fig_dist = plot_label_distribution(st.session_state.df, target_column)
            st.pyplot(fig_dist)
        with eda_cols[1]:
            st.subheader("Word Cloud (from original text)")
            fig_wc = generate_wordcloud(st.session_state.df, text_column)
            st.pyplot(fig_wc)
        
        st.subheader("N-gram Analysis")
        n_gram = st.slider("Select N for N-grams", 2, 5, 2)
        top_k = st.slider("Select Top K results", 10, 50, 20)
        fig_ngram = plot_top_ngrams(st.session_state.df, text_column, n=n_gram, top_k=top_k)
        st.pyplot(fig_ngram)


    with tabs[1]:
        st.header("Latest Model Results")
        if not st.session_state.model_results:
            st.warning("No experiments have been run yet.")
        else:
            latest_result = st.session_state.model_results[-1]
            st.subheader(f"Results for: {latest_result['model_name']}")

            metric_cols = st.columns(2)
            metric_cols[0].metric("Accuracy Score", f"{latest_result['accuracy']:.4f}")
            metric_cols[1].metric("F1-Score (Weighted)", f"{latest_result['f1_score']:.4f}")

            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix(latest_result['y_test'], latest_result['y_pred'])
            st.pyplot(fig_cm)
            
            st.header("Export")
            export_cols = st.columns(2)
            # Download Model
            pkl_model = pickle.dumps(latest_result['trained_model'])
            b64_pkl = base64.b64encode(pkl_model).decode()
            export_cols[0].download_button(
                label="Download Model (.pkl)",
                data=pkl_model,
                file_name=f"{latest_result['model_name']}.pkl",
                mime="application/octet-stream"
            )
            # Download Report
            report_df = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score'],
                'Score': [latest_result['accuracy'], latest_result['f1_score']]
            })
            csv_report = report_df.to_csv(index=False).encode('utf-8')
            export_cols[1].download_button(
                label="Download Report (.csv)",
                data=csv_report,
                file_name=f"{latest_result['model_name']}_report.csv",
                mime="text/csv"
            )

    with tabs[2]:
        st.header("Model Comparison")
        if not st.session_state.model_results:
            st.warning("Run multiple experiments to compare models here.")
        else:
            comparison_df = pd.DataFrame(st.session_state.model_results).drop(
                columns=['trained_model', 'y_test', 'y_pred']
            )
            st.dataframe(comparison_df)
            if st.button("Clear All Results"):
                st.session_state.model_results = []
                st.rerun()