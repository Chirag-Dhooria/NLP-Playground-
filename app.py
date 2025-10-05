import streamlit as st
import pandas as pd
import pickle
import base64
from utils.data_loader import load_data
from utils.preprocessing import preprocess_text
from utils.visualizer import plot_label_distribution, generate_wordcloud, plot_confusion_matrix, plot_top_ngrams
from utils.model_handler import get_model, train_model, evaluate_model


st.set_page_config(page_title="NLP Playground", page_icon="🛝", layout="wide")


if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = []


st.title("NLP Playground 🛝")
st.markdown("""
Welcome to the NLP Playground! This is a no-code platform designed to help you explore, preprocess, and analyze your text data. 
You can train classical machine learning models for text classification and compare their performance, all without writing a single line of code.
""")

st.markdown("### How It Works:")
st.markdown("""
1.  **Upload Your Data**: Use the sidebar to upload your text dataset in CSV or JSON format.
2.  **Configure Your Experiment**: Select your text and label columns, choose your desired preprocessing steps, and pick a model to train.
3.  **Run & Analyze**: Click the "Run Experiment" button. The results, including performance metrics and visualizations, will appear in the main panel.
4.  **Compare**: Run multiple experiments with different settings and compare the results in the "Model Comparison" tab.
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
            'lowercase': st.checkbox("Convert to Lowercase", value=True, help="Converts all text to lowercase letters."),
            'remove_punctuation': st.checkbox("Remove Punctuation", value=True, help="Removes all punctuation characters from the text."),
            'remove_stopwords': st.checkbox("Remove Stopwords", help="Removes common English words that don't add much meaning (e.g., 'the', 'a', 'is')."),
            'lemmatization': st.checkbox("Lemmatization", help="Reduces words to their base or dictionary form (e.g., 'running' -> 'run'). Slower but more accurate than stemming."),
            'stemming': st.checkbox("Stemming", help="Reduces words to their root form (e.g., 'running' -> 'run'). Faster but less accurate than lemmatization.")
        }

        st.header("4. Model Selection")
        model_name = st.selectbox(
            "Choose a Model",
            ["Logistic Regression", "Naive Bayes", "Support Vector Machine (SVM)", "Random Forest", "Gradient Boosting"]
        )
        if model_name == "Logistic Regression":
            st.info("A simple and efficient linear model for binary and multiclass classification.")
        elif model_name == "Naive Bayes":
            st.info("A probabilistic classifier based on Bayes' theorem, works well with text data.")
        elif model_name == "Support Vector Machine (SVM)":
            st.info("A powerful model that finds the optimal hyperplane to separate classes.")
        elif model_name == "Random Forest":
            st.info("An ensemble model using multiple decision trees to improve accuracy and control overfitting.")
        elif model_name == "Gradient Boosting":
            st.info("An ensemble technique that builds models sequentially, each one correcting the errors of its predecessor.")

        
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

        if st.button("🚀 Run Experiment", use_container_width=True):
            with st.status("Running experiment...", expanded=True) as status:
                st.write("Step 1: Preprocessing data...")
                processed_df = preprocess_text(df.copy(), text_column, options)
                
                status.update(label="Step 2: Training model...")
                model = get_model(model_name)
                
                trained_model, vectorizer, X_test_vec, y_test = train_model(
                    processed_df, 'processed_text', target_column, model, test_size
                )
                
                status.update(label="Step 3: Evaluating performance...")
                metrics, y_pred = evaluate_model(trained_model, X_test_vec, y_test)
                
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

                if len(st.session_state.model_results) > 10:
                    st.session_state.model_results.pop(0)

                status.update(label="Experiment complete!", state="complete", expanded=False)

            st.success("Experiment finished successfully!")


if st.session_state.df is None:
    st.info("Upload a dataset using the sidebar to begin.")
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
            metric_cols[1].metric("F1-Score (Weighted)", f"{latest_result['f1_score']:.4f}", help="The F1-score, weighted by the number of true instances for each label. It's a useful metric for datasets with an imbalanced class distribution.")

            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix(latest_result['y_test'], latest_result['y_pred'])
            st.pyplot(fig_cm)
            
            st.header("Export")
            export_cols = st.columns(2)
            pkl_model = pickle.dumps(latest_result['trained_model'])
            export_cols[0].download_button(
                label="Download Model (.pkl)",
                data=pkl_model,
                file_name=f"{latest_result['model_name']}.pkl",
                mime="application/octet-stream"
            )
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
        st.markdown("Here you can compare the results of all experiments from this session.")
        
        if not st.session_state.model_results:
            st.warning("Run multiple experiments to compare models here.")
        else:
            
            results_list = []
            for result in st.session_state.model_results:
                results_list.append({
                    "Model": result['model_name'],
                    "Accuracy": f"{result['accuracy']:.4f}",
                    "F1-Score": f"{result['f1_score']:.4f}",
                    "Preprocessing Steps": result['preprocessing']
                })
            comparison_df = pd.DataFrame(results_list)
            st.dataframe(comparison_df)

            # Display results 
            st.markdown("---")
            st.subheader("Download Artifacts for Each Run")
            
            for i, result in enumerate(reversed(st.session_state.model_results)):
                expander_title = f"**{result['model_name']}** (Accuracy: {result['accuracy']:.4f})"
                with st.expander(expander_title):
                    st.write(f"**Preprocessing:** {result['preprocessing']}")
                    st.write(f"**F1-Score:** {result['f1_score']:.4f}")
                    
                    download_cols = st.columns(2)
                    
                    # Download Model (.pkl)
                    pkl_model = pickle.dumps(result['trained_model'])
                    download_cols[0].download_button(
                        label="Download Model",
                        data=pkl_model,
                        file_name=f"model_{result['model_name']}_{i}.pkl",
                        mime="application/octet-stream",
                        key=f"pkl_{i}",
                        use_container_width=True
                    )

                    # Download Report (.csv)
                    report_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'F1 Score'],
                        'Score': [result['accuracy'], result['f1_score']]
                    })
                    csv_report = report_df.to_csv(index=False).encode('utf-8')
                    download_cols[1].download_button(
                        label="Download Report",
                        data=csv_report,
                        file_name=f"report_{result['model_name']}_{i}.csv",
                        mime="text/csv",
                        key=f"csv_{i}",
                        use_container_width=True
                    )

            if st.button("Clear All Results", use_container_width=True):
                st.session_state.model_results = []
                st.rerun()